"""
R2.1 + R2.2 runner.

  uv run python -u src/compress/experimental/run_geo_thermal.py models/cylinder_graph/<run> \
      --nbeta-scales 1 0.1 0.01 --gammas 100 10 --eps 3e-7 --steps 2000 --burn-in 500 --thin 2 \
      --n-probes 256 --n-calib 1024 --threads 8 --out results/compression/geometry/

Outputs: r2_nongauss.json  (per temperature: chain summary, per-tensor rho table, plateau check)
         r2_alloc.csv      (posterior-variance allocation rows, PTQ and +QAT200 soft)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.base import is_quantized_param
from compress.experimental.geo_alloc import (MENU_DEFAULT, ExtGPTQ, ExtSTEQAT, count_bytes_ext,
                                             knapsack, width_label)
from compress.experimental.geo_thermal import hutchinson_diag, nongauss_map, posterior_omega, sgld_moments
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--nbeta-scales", nargs="+", type=float, default=[1.0, 0.1, 0.01])
    ap.add_argument("--gammas", nargs="+", type=float, default=[100.0, 10.0],
                    help="first gamma used at every scale; further gammas only at scale 1")
    ap.add_argument("--eps", type=float, default=3e-7)
    ap.add_argument("--eps-per-scale", nargs="+", type=float, default=None,
                    help="optional eps for each entry of --nbeta-scales (default: --eps for all). "
                         "SGLD's OU relaxation time in a direction of curvature h is ~2/(eps (nbeta h + gamma)) "
                         "steps, so hotter chains may use eps scaled by 1/scale to keep eps*nbeta fixed.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-probes", type=int, default=256)
    ap.add_argument("--n-calib", type=int, default=1024)
    ap.add_argument("--budgets", nargs="+", type=int, default=[95744, 82944, 70272, 60000])
    ap.add_argument("--qat-steps", type=int, default=200)
    ap.add_argument("--n-gptq-calib", type=int, default=128)
    ap.add_argument("--e4-sensitivity", default="results/compression/geometry/e4_sensitivity.json")
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tensors-print", type=int, default=100)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    menu = MENU_DEFAULT

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, max(args.n_calib, 4096), seed=0)
    hcalib = calib[: args.n_calib]
    gptq_calib = calib[: args.n_gptq_calib]
    p_test, p_calib, p_h = teacher_probs(proc, test), teacher_probs(proc, calib), teacher_probs(proc, hcalib)
    base_t, base_c = evaluate_teacher(model, test, p_test), evaluate_teacher(model, calib, p_calib)
    names = [n for n, _ in model.named_parameters() if is_quantized_param(n)]
    params = dict(model.named_parameters())
    n = len(train)
    nbeta0 = 16 * n / math.log(n)
    print(f"fp32 test nll {base_t['nll']:.4f} kl {base_t['kl']:.5f} | calib kl {base_c['kl']:.5f} | nbeta_default {nbeta0:.3e}", flush=True)

    measured = None
    if Path(args.e4_sensitivity).exists():
        S = json.load(open(args.e4_sensitivity))
        measured = {nm: {int(b): v for b, v in d.items()} for nm, d in S["measured"].items()}

    # ---- Hessian diagonal (temperature independent) ----------------------------------------
    t0 = time.time()
    diag_h = hutchinson_diag(model, hcalib, p_h, n_probes=args.n_probes, seed=0)
    print(f"hutchinson diag: {args.n_probes} probes on {len(hcalib)} seqs [{time.time()-t0:.0f}s]; "
          f"per-tensor mean h: " + ", ".join(f"{nm.split('.')[-1]}{nm.split('.')[1] if nm.startswith('blocks') else ''}={float(diag_h[nm].mean()):.2e}"
                                             for nm in names[:6]) + " ...", flush=True)

    # ---- byte costs (as in run_geo_alloc.py) --------------------------------------------
    base_bytes = count_bytes_ext(model, {})
    cost = {nm: {b: count_bytes_ext(model, {nm: b}) - base_bytes + params[nm].numel() * 4 for b in menu} for nm in names}
    bias_bytes = sum(p.numel() * 4 for nm, p in model.named_parameters() if not is_quantized_param(nm))

    combos = []
    eps_map = {}
    for i, s in enumerate(args.nbeta_scales):
        eps_map[s] = args.eps_per_scale[i] if args.eps_per_scale else args.eps
        gl = args.gammas if s == 1.0 else args.gammas[:1]
        for gm in gl:
            combos.append((s, gm))

    results = {"nbeta_default": nbeta0, "eps": args.eps, "steps": args.steps, "burn_in": args.burn_in,
               "thin": args.thin, "base_calib_kl": base_c["kl"], "temperatures": {}}
    rows = []
    for s, gm in combos:
        nb = s * nbeta0
        key = f"scale{s:g}|gamma{gm:g}"
        t0 = time.time()
        eps = eps_map[s]
        ch = sgld_moments(model, train, proc, nb, gm, eps, args.steps, args.burn_in,
                          batch_size=args.batch_size, seed=0, thin=args.thin, loss_star=base_c["kl"])
        # plateau check: relative change of tracked variances between mid and end
        pl = ch["plateau"]
        plateaued = None
        if len(pl) >= 2:
            ks = sorted(pl, key=int)
            a, b = t.tensor(pl[ks[-2]]), t.tensor(pl[ks[-1]])
            rel = float(((b - a).abs() / (a.abs() + 1e-20)).median())
            plateaued = dict(median_rel_change_mid_to_end=rel, ok=rel < 0.2)
        print(f"[{key}] nbeta={nb:.3e} eps={eps:g} T={1/nb:.2e}  dL=<L>-L*={1e3*ch['delta_loss']:+.3f} mnat  lambda={ch['lambda_hat']:.1f}  "
              f"samples={ch['n_samples']} diverged={ch['diverged']}  |w-w*|={ch['dist_trace'][-1] if ch['dist_trace'] else float('nan'):.2f}  "
              f"plateau={plateaued}  [{time.time()-t0:.0f}s]", flush=True)
        rec = dict(nbeta=nb, gamma=gm, eps=eps, T=1 / nb, delta_loss=ch["delta_loss"], lambda_hat=ch["lambda_hat"],
                   n_samples=ch["n_samples"], diverged=ch["diverged"], plateau=plateaued,
                   plateau_raw=pl, loss_trace=ch["loss_trace"], dist_trace=ch["dist_trace"])
        if ch["diverged"]:
            results["temperatures"][key] = rec
            with open(out / "r2_nongauss.json", "w") as f:
                json.dump(results, f, indent=1)
            continue

        # ---- R2.1 map -----------------------------------------------------------------
        nm_map = nongauss_map(ch["var"], diag_h, nb, gm)
        rec["nongauss"] = nm_map
        print(f"  {'tensor':24s} {'median rho':>10s} {'IQR':>16s} {'f<0.3':>6s} {'f>3':>6s} {'mean var':>10s} {'mean h':>10s} {'tensor rho':>10s}")
        for nm in names[: args.tensors_print]:
            r = nm_map[nm]
            print(f"  {nm:24s} {r['median']:10.3f} [{r['q25']:6.3f},{r['q75']:7.3f}] {r['frac_lt_0p3']:6.2f} {r['frac_gt_3']:6.2f} "
                  f"{r['mean_var']:10.3e} {r['mean_h']:10.3e} {r['tensor_rho']:10.3f}")
        allrho = t.cat([(ch["var"][nm] / ((1 / nb) / (diag_h[nm].clamp_min(0) + gm / nb))).reshape(-1) for nm in names])
        rec["global"] = dict(median=float(allrho.median()), frac_lt_0p3=float((allrho < 0.3).float().mean()),
                             frac_gt_3=float((allrho > 3).float().mean()),
                             frac_within_3x=float(((allrho > 1 / 3) & (allrho < 3)).float().mean()))
        print(f"  global: median rho {rec['global']['median']:.3f}, within 3x: {rec['global']['frac_within_3x']:.2f}, "
              f"<0.3: {rec['global']['frac_lt_0p3']:.2f}, >3: {rec['global']['frac_gt_3']:.2f}", flush=True)
        results["temperatures"][key] = rec
        with open(out / "r2_nongauss.json", "w") as f:
            json.dump(results, f, indent=1)

        # ---- R2.2 allocation ------------------------------------------------------------
        omega = posterior_omega(model, ch["var"], nb, menu)
        for budget in args.budgets:
            try:
                alloc = knapsack(names, omega, cost, budget - bias_bytes, menu)
            except AssertionError:
                print(f"  budget {budget}: infeasible"); continue
            q = ExtGPTQ(16, n_calib=args.n_gptq_calib); q.bits_map = alloc
            qm = q.quantize(model, gptq_calib)
            rt = evaluate_teacher(qm, test, p_test)
            actual = count_bytes_ext(qm, alloc)
            pred = sum(measured[nm][alloc[nm]] for nm in names) if measured else None
            hist = {}
            for b in alloc.values():
                hist[width_label(b)] = hist.get(width_label(b), 0) + 1
            row = dict(temperature_key=key, budget=budget, actual_bytes=actual, stage="PTQ",
                       test_nll=rt["nll"], test_kl=rt["kl"], additive_pred_measured=pred,
                       hist=json.dumps(hist), allocation=json.dumps({nm: width_label(b) for nm, b in alloc.items()}))
            rows.append(row)
            print(f"  alloc budget={budget:6d} bytes={actual:6d} PTQ test_nll={rt['nll']:.4f} (d={1e3*(rt['nll']-base_t['nll']):+.1f} mnat)"
                  f"  additive(measured)={1e3*pred if pred is not None else float('nan'):.1f}  {hist}", flush=True)
            if args.qat_steps:
                qq = ExtSTEQAT(16, max_steps=args.qat_steps, soft_targets=True, process=proc)
                qq.bits_map = alloc
                qm2 = qq.quantize(qm, train)
                rt2 = evaluate_teacher(qm2, test, p_test)
                rows.append(dict(row, stage=f"QAT{args.qat_steps}_soft", test_nll=rt2["nll"], test_kl=rt2["kl"]))
                print(f"         +QAT{args.qat_steps} soft test_nll={rt2['nll']:.4f} (d={1e3*(rt2['nll']-base_t['nll']):+.1f} mnat)", flush=True)
            with open(out / "r2_alloc.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
