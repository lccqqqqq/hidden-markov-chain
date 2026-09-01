"""
E3 runner: per checkpoint -> test loss, RTN knee, Lanczos spectral summary, SGLD LLC.

  uv run python -u src/compress/experimental/run_geo_llc.py \
      --runs models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN models/cylinder_graph/20260828_140456_L4_d32_H2_full_noLN \
      --ckpts best_model.pt checkpoint_epoch_1.pt ... --out results/compression/geometry/e3_llc.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.base import count_params
from compress.experimental.geo_llc import FlatHVP, sgld_llc, slq_summary
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from compress.rtn import RTN
from evaluate import evaluate, load_run, load_test_data
from utils import create_process_from_dict


def knee(bits, dloss, thr):
    """Largest bit width at which dloss first exceeds thr, linearly interpolated (in bits)."""
    pts = sorted(zip(bits, dloss), reverse=True)     # descending bits
    for (b1, d1), (b0, d0) in zip(pts, pts[1:]):
        if d0 >= thr > d1:
            return b1 - (thr - d1) / (d0 - d1) * (b1 - b0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", default=["best_model.pt"] + [f"checkpoint_epoch_{i}.pt" for i in range(1, 11)])
    ap.add_argument("--n-calib", type=int, default=2048)
    ap.add_argument("--lanczos-m", type=int, default=60)
    ap.add_argument("--lanczos-probes", type=int, default=4)
    ap.add_argument("--gammas", nargs="+", type=float, default=[100.0, 1000.0])
    ap.add_argument("--nbeta-scale", nargs="+", type=float, default=[1.0],
                    help="multiples of the default nbeta_token = 16 n / log n")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--skip-sgld", action="store_true")
    ap.add_argument("--skip-lanczos", action="store_true")
    ap.add_argument("--out", default="results/compression/geometry/e3_llc.json")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    t.set_num_threads(args.threads)

    results = []
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        for ck in args.ckpts:
            t0 = time.time()
            model, cfg = load_run(run, ck)
            proc = create_process_from_dict(cfg["data_generator"]["process"])
            test = load_test_data(cfg, "test")
            train = load_test_data(cfg, "train")
            calib = calib_split(train, args.n_calib, seed=0)
            p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
            n = len(train)
            nbeta0 = 16 * n / math.log(n)
            rec = dict(run=Path(run).name, ckpt=ck, params=count_params(model), n_train=n, nbeta_token_default=nbeta0)
            rec.update({f"test_{k}": v for k, v in evaluate_teacher(model, test, p_test).items() if k in ("nll", "kl")})
            rec["calib_kl"] = evaluate_teacher(model, calib, p_calib)["kl"]

            # RTN knee (test NLL, per-channel asym)
            bits = [8, 6, 5, 4, 3, 2]
            dl = []
            for b in bits:
                qm = RTN(b).quantize(model)
                dl.append(evaluate(qm, test)["loss"] - rec["test_nll"])
            rec["rtn_bits"], rec["rtn_dloss"] = bits, dl
            rec["knee_10mnat"], rec["knee_50mnat"] = knee(bits, dl, 0.010), knee(bits, dl, 0.050)

            nbetas = [s * nbeta0 for s in args.nbeta_scale]
            if not args.skip_lanczos:
                hvp = FlatHVP(model, calib, p_calib)
                rec["slq"] = slq_summary(hvp, args.lanczos_m, args.lanczos_probes, nbetas, args.gammas, seed=0)
                del hvp
            if not args.skip_sgld:
                rec["sgld"] = {}
                for nb in nbetas:
                    for gm in args.gammas:
                        chains = [sgld_llc(model, train, proc, nb, gm, args.eps, args.steps, args.burn_in,
                                           seed=s, loss_star=rec["calib_kl"]) for s in range(args.chains)]
                        lams = [c["lambda_hat"] for c in chains]
                        rec["sgld"][f"{nb:.3e}|{gm:g}"] = dict(
                            lambda_hat=float(np.mean(lams)), lambda_std=float(np.std(lams)),
                            diverged=any(c["diverged"] for c in chains),
                            final_dist=[c["dist_trace"][-1] if c["dist_trace"] else None for c in chains],
                            loss_trace=chains[0]["loss_trace"])
            results.append(rec)
            lq = rec.get("slq", {}).get("lambda_quad", {})
            ls = {k: v["lambda_hat"] for k, v in rec.get("sgld", {}).items()}
            print(f"{rec['run'][-22:]} {ck:22s} nll={rec['test_nll']:.4f} kl={rec['test_kl']:.5f} "
                  f"knee10={rec['knee_10mnat']:.2f}b  lambda_quad={ {k: round(v,1) for k,v in lq.items()} } "
                  f"lambda_sgld={ {k: round(v,1) for k,v in ls.items()} }  [{time.time()-t0:.0f}s]", flush=True)
            with open(out, "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
