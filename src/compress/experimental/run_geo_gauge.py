"""
E10 runner: canonicalize (exact gauge transforms) -> reference RTN / GPTQ at bits {4,3}.

  uv run python -u src/compress/experimental/run_geo_gauge.py \
      models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN \
      --modes none relu_equalize qk_balance vo_balance qk_rotate vo_rotate all \
      --bits 4 3 --out results/compression/geometry/e10_gauge.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_gauge import MODES, canonicalize
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from compress.gptq import GPTQ
from compress.rtn import RTN
from evaluate import load_run, load_test_data
from utils import create_process_from_dict

COLS = ["mode", "quantizer", "bits", "test_nll", "test_kl", "delta_vs_no_gauge_mnat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3])
    ap.add_argument("--quantizers", nargs="+", default=["RTN", "GPTQ"])
    ap.add_argument("--n-gptq-calib", type=int, default=128)
    ap.add_argument("--rot-steps", type=int, default=200)
    ap.add_argument("--out", default="results/compression/geometry/e10_gauge.csv")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    p_test = teacher_probs(proc, test)
    calib = calib_split(train, max(256, args.n_gptq_calib), seed=0)
    gptq_calib = calib[: args.n_gptq_calib]
    base = evaluate_teacher(model, test, p_test)
    print(f"fp32 test nll={base['nll']:.4f} kl={base['kl']:.5f}", flush=True)

    rows, ref = [], {}
    for mode in args.modes:
        t0 = time.time()
        cm = canonicalize(model, mode, check_tokens=calib, rot_steps=args.rot_steps)
        chk = evaluate_teacher(cm, test, p_test)
        print(f"[{mode}] fp32 after gauge: nll={chk['nll']:.5f} (Δ={1e6*(chk['nll']-base['nll']):+.1f} µnat) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        for qn in args.quantizers:
            for b in args.bits:
                q = RTN(b) if qn == "RTN" else GPTQ(b, n_calib=args.n_gptq_calib)
                qm = q.quantize(cm, gptq_calib)
                r = evaluate_teacher(qm, test, p_test)
                if mode == "none":
                    ref[(qn, b)] = r["nll"]
                d = 1e3 * (r["nll"] - ref[(qn, b)]) if (qn, b) in ref else float("nan")
                rows.append(dict(mode=mode, quantizer=qn, bits=b, test_nll=r["nll"], test_kl=r["kl"],
                                 delta_vs_no_gauge_mnat=round(d, 3)))
                print(f"  {qn:4s} b={b}  nll={r['nll']:.4f} (d vs fp32 {1e3*(r['nll']-base['nll']):+.1f} mnat; "
                      f"vs no-gauge {d:+.2f} mnat)", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

    print("\nmode".ljust(15) + "".join(f"{qn}-{b}".rjust(11) for qn in args.quantizers for b in args.bits))
    for mode in args.modes:
        line = mode.ljust(14)
        for qn in args.quantizers:
            for b in args.bits:
                v = [r for r in rows if r["mode"] == mode and r["quantizer"] == qn and r["bits"] == b][0]
                line += f"{v['delta_vs_no_gauge_mnat']:+11.2f}"
        print(line)


if __name__ == "__main__":
    main()
