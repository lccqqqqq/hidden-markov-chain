"""
E9 runner.

  uv run python -u src/compress/experimental/run_geo_tie.py models/cylinder_graph/<run> \
      --out results/compression/geometry/

Configurations: (n_shared, n_apply) ∈ {(1,4), (2,4), (1,8)} x init ∈ {mean, layer0, best_single(1,4 only)}
x recovery ∈ {200 steps, 1 epoch}. best_single: each trained layer used ×4, 200-step recovery,
the best one is then also given the 1-epoch recovery.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_recover import recover, recovery_label
from compress.experimental.geo_tie import build_tied, check_tied, unique_bytes_fp32, unique_params
from compress.experimental.teacher import evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict

FIELDS = ["n_shared", "n_apply", "init", "unique_params", "bytes_fp32", "recovery", "test_nll", "test_kl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--configs", nargs="+", default=["1x4", "2x4", "1x8"], help="n_shared x n_apply")
    ap.add_argument("--inits", nargs="+", default=["mean", "layer0", "best_single"])
    ap.add_argument("--short-steps", type=int, default=200)
    ap.add_argument("--no-epoch", action="store_true")
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
    base = evaluate_teacher(model, test, p_test)
    print(f"fp32: nll={base['nll']:.4f} kl={base['kl']:.5f}", flush=True)
    rows = []

    def run(n_shared, n_apply, init, steps, source_layer=None, label=None):
        m = build_tied(model, cfg["model"], n_shared, n_apply, init, source_layer)
        assert check_tied(m, n_shared)
        t1 = time.time()
        m = recover(m, train, proc, steps, soft_targets=args.soft_targets)
        r = evaluate_teacher(m, test, p_test)
        row = dict(n_shared=n_shared, n_apply=n_apply, init=label or init, unique_params=unique_params(m),
                   bytes_fp32=unique_bytes_fp32(m), recovery=recovery_label(steps), test_nll=r["nll"], test_kl=r["kl"])
        rows.append(row)
        print(f"{n_shared}x{n_apply} {row['init']:12s} params={row['unique_params']:6d} {row['recovery']:9s} "
              f"nll={r['nll']:.4f} (d={1e3*(r['nll']-base['nll']):+.1f} mnat) kl={r['kl']:.5f} [{time.time()-t1:.0f}s]", flush=True)
        with open(out / "e9_tie.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        return r["nll"]

    recs = [args.short_steps] + ([] if args.no_epoch else [None])
    for c in args.configs:
        n_shared, n_apply = map(int, c.split("x"))
        for init in args.inits:
            if init == "best_single":
                if n_shared != 1 or n_apply != 4:
                    continue
                scores = {l: run(1, 4, "single", args.short_steps, l, f"single_L{l}") for l in range(model.cfg.n_layers)}
                best = min(scores, key=scores.get)
                if not args.no_epoch:
                    run(1, 4, "single", None, best, f"best_single_L{best}")
                continue
            for steps in recs:
                run(n_shared, n_apply, init, steps)


if __name__ == "__main__":
    main()
