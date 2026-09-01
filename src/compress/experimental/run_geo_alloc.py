"""
E4 runner.

  uv run python -u src/compress/experimental/run_geo_alloc.py models/cylinder_graph/<run> \
      --budgets 95744 82944 70272 60000 50000 40000 --out results/compression/geometry/

Outputs: e4_sensitivity.json (S_i(b) measured + Omega_i(b) trace-based, on the same menu)
         e4_alloc.csv        (one row per budget x sensitivity signal x {PTQ, +QAT200})
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.base import Quantizer, is_quantized_param
from compress.experimental.geo_alloc import (DELETE, MENU_DEFAULT, TERNARY, ExtGPTQ, ExtSTEQAT,
                                             count_bytes_ext, knapsack, ternary_fq, width_label)
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from compress.hawq import HAWQv2
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--budgets", nargs="+", type=int, default=[95744, 82944, 70272, 60000, 50000, 40000])
    ap.add_argument("--menu", nargs="+", type=int, default=list(MENU_DEFAULT))
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--n-gptq-calib", type=int, default=128)
    ap.add_argument("--qat-steps", type=int, default=200)
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--reuse-sensitivity", action="store_true")
    ap.add_argument("--signals", nargs="+", default=["measured", "trace"])
    ap.add_argument("--qat-soft-only", action="store_true", help="skip the one-hot QAT row")
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    menu = tuple(args.menu)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    gptq_calib = calib[: args.n_gptq_calib]
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
    base_c, base_t = evaluate_teacher(model, calib, p_calib), evaluate_teacher(model, test, p_test)
    names = [n for n, _ in model.named_parameters() if is_quantized_param(n)]
    params = dict(model.named_parameters())

    # ---- costs ------------------------------------------------------------------------------
    cost = {n: {b: count_bytes_ext(model, {n: b}) - count_bytes_ext(model, {}) + 0 for b in menu} for n in names}
    # count_bytes_ext(model, {n: b}) counts all other tensors at fp32; the difference above is
    # (bytes of n at b) - (bytes of n at fp32); shift to absolute per-tensor bytes:
    for n in names:
        fp32 = params[n].numel() * 4
        cost[n] = {b: fp32 + cost[n][b] for b in menu}
    bias_bytes = sum(p.numel() * 4 for n, p in model.named_parameters() if not is_quantized_param(n))

    # ---- sensitivities ----------------------------------------------------------------------
    sens_path = out / "e4_sensitivity.json"
    if args.reuse_sensitivity and sens_path.exists():
        S = json.load(open(sens_path))
        S = {k: {n: {int(b): v for b, v in d.items()} for n, d in S[k].items()} for k in ("measured", "trace")}
    else:
        t0 = time.time()
        measured = {n: {} for n in names}
        for n in names:
            for b in menu:
                q = ExtGPTQ(16, n_calib=args.n_gptq_calib)
                q.bits_map = {m: 16 for m in names}; q.bits_map[n] = b
                qm = q.quantize(model, gptq_calib)
                measured[n][b] = evaluate_teacher(qm, calib, p_calib)["kl"] - base_c["kl"]
            print(f"S[{n:24s}] " + " ".join(f"{width_label(b)}:{measured[n][b]*1e3:7.2f}" for b in menu)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)
        # trace-based Omega on the same menu (HAWQ-V2's formula; traces from the reference code)
        h = HAWQv2(0, ExtGPTQ(16), choices=(2, 4, 8))
        traces = h.hessian_traces(model, gptq_calib)
        trace_o = {n: {} for n in names}
        for n in names:
            w2 = Quantizer._channel_view(params[n].data, "per_channel", n)
            tr = max(traces[n], 0.0)
            for b in menu:
                if b == DELETE:
                    err = w2.pow(2).sum().item()
                elif b == TERNARY:
                    err = (ternary_fq(w2)[0] - w2).pow(2).sum().item()
                else:
                    s_, z_, lo, hi = Quantizer.grid(w2, b, False)
                    err = (Quantizer.fake_quant(w2, s_, z_, lo, hi) - w2).pow(2).sum().item()
                trace_o[n][b] = tr / params[n].numel() * err
        S = {"measured": measured, "trace": trace_o, "traces": traces, "base_calib_kl": base_c["kl"]}
        with open(sens_path, "w") as f:
            json.dump(S, f, indent=1)

    # ---- allocate, apply, evaluate ----------------------------------------------------------
    rows = []
    for signal in args.signals:
        value = S[signal]
        for budget in args.budgets:
            try:
                alloc = knapsack(names, value, cost, budget - bias_bytes, menu)
            except AssertionError:
                print(f"{signal} budget {budget}: infeasible"); continue
            q = ExtGPTQ(16, n_calib=args.n_gptq_calib); q.bits_map = alloc
            qm = q.quantize(model, gptq_calib)
            rc, rt = evaluate_teacher(qm, calib, p_calib), evaluate_teacher(qm, test, p_test)
            actual = count_bytes_ext(qm, alloc)
            pred = sum(S["measured"][n][alloc[n]] for n in names)       # additive prediction (measured)
            hist = {}
            for b in alloc.values():
                hist[width_label(b)] = hist.get(width_label(b), 0) + 1
            row = dict(signal=signal, budget=budget, actual_bytes=actual, stage="PTQ",
                       test_nll=rt["nll"], test_kl=rt["kl"], calib_kl=rc["kl"],
                       additive_pred_dkl=pred, joint_dkl=rc["kl"] - base_c["kl"],
                       hist=json.dumps(hist), allocation=json.dumps({n: width_label(b) for n, b in alloc.items()}))
            rows.append(row)
            print(f"{signal:8s} budget={budget:6d} bytes={actual:6d} PTQ test_nll={rt['nll']:.4f} "
                  f"(d={1e3*(rt['nll']-base_t['nll']):+.1f} mnat)  additive={1e3*pred:.1f} joint={1e3*row['joint_dkl']:.1f}  {hist}", flush=True)
            if args.qat_steps:
                for soft in ((True,) if args.qat_soft_only else (False, True)):
                    qq = ExtSTEQAT(16, max_steps=args.qat_steps, soft_targets=soft, process=proc)
                    qq.bits_map = alloc
                    # start from the GPTQ-rounded model (allocation -> rounding -> short recovery)
                    qm2 = qq.quantize(qm, train)
                    rt2 = evaluate_teacher(qm2, test, p_test)
                    rows.append(dict(row, stage=f"QAT{args.qat_steps}{'_soft' if soft else ''}",
                                     test_nll=rt2["nll"], test_kl=rt2["kl"], calib_kl=None,
                                     additive_pred_dkl=None, joint_dkl=None))
                    print(f"         +QAT{args.qat_steps}{' soft' if soft else '     '} test_nll={rt2['nll']:.4f} "
                          f"(d={1e3*(rt2['nll']-base_t['nll']):+.1f} mnat)", flush=True)
            with open(out / "e4_alloc.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
