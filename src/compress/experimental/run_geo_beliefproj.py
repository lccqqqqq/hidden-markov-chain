"""
E8 runner.

  uv run python -u src/compress/experimental/run_geo_beliefproj.py models/cylinder_graph/<run> \
      --dprimes 18 24 32 --out results/compression/geometry/

Rows: subspace ∈ {probe, pca, rand} x d' x recovery ∈ {0, 200 steps, (1 epoch: probe only)}.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.base import count_bytes, count_params
from compress.experimental.geo_beliefproj import (collect_residuals, compile_projected,
                                                  fit_belief_probes, subspace)
from compress.experimental.geo_recover import recover, recovery_label
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dprimes", nargs="+", type=int, default=[18, 24, 32])
    ap.add_argument("--subspaces", nargs="+", default=["probe", "pca", "rand"])
    ap.add_argument("--recover-steps", nargs="+", type=int, default=[0, 200])
    ap.add_argument("--epoch-for", nargs="+", default=["probe"], help="subspaces that also get a 1-epoch row")
    ap.add_argument("--n-probe", type=int, default=8192)
    ap.add_argument("--soft-targets", action="store_true")
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    p_test = teacher_probs(proc, test)
    probe_tr = calib_split(train, args.n_probe + 2048, seed=1)
    probe_tr, probe_ho = probe_tr[: args.n_probe], probe_tr[args.n_probe:]

    base = evaluate_teacher(model, test, p_test)
    print(f"fp32: params={count_params(model)} nll={base['nll']:.4f} kl={base['kl']:.5f}", flush=True)

    t0 = time.time()
    probe_W, probe_r2 = fit_belief_probes(model, proc, probe_tr, probe_ho)
    print("probe R² per read point:", {k: round(v, 4) for k, v in probe_r2.items()}, f"[{time.time()-t0:.0f}s]", flush=True)
    resid = collect_residuals(model, probe_tr[:2048])

    rows = []
    fields = ["subspace", "d_prime", "params", "bytes_fp32", "recovery", "test_nll", "test_kl", "probe_r2"]
    for d_prime in args.dprimes:
        for kind in args.subspaces:
            P = subspace(kind, d_prime, probe_W=probe_W, resid=resid, d_model=model.cfg.d_model)
            compiled = compile_projected(model, cfg["model"], P)
            recs = list(args.recover_steps) + ([None] if kind in args.epoch_for else [])
            for steps in recs:
                m = copy.deepcopy(compiled)
                t1 = time.time()
                m = recover(m, train, proc, steps, soft_targets=args.soft_targets)
                r = evaluate_teacher(m, test, p_test)
                row = dict(subspace=kind, d_prime=d_prime, params=count_params(m), bytes_fp32=count_bytes(m, None),
                           recovery=recovery_label(steps), test_nll=r["nll"], test_kl=r["kl"],
                           probe_r2=json.dumps({k: round(v, 4) for k, v in probe_r2.items()}) if kind == "probe" else "")
                rows.append(row)
                print(f"{kind:6s} d'={d_prime:2d} params={row['params']:6d} {row['recovery']:9s} "
                      f"nll={r['nll']:.4f} (d={1e3*(r['nll']-base['nll']):+.1f} mnat) kl={r['kl']:.5f} [{time.time()-t1:.0f}s]", flush=True)
                with open(out / "e8_beliefproj.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
