"""
HAWQ-V2 mixed-precision runs at byte budgets matching the uniform-bit rows, over RTN and
GPTQ bases, so the mixed and uniform rows are comparable at byte parity.

  uv run python src/compress/run_hawq.py models/cylinder_graph/<run> \
      --budgets 121216 95744 70272 --bases rtn gptq --out results/compression/<name>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.base import count_bytes, count_params
from compress.gptq import GPTQ
from compress.hawq import HAWQv2
from compress.rtn import RTN
from evaluate import evaluate, load_run, load_test_data

BASES = {"rtn": lambda: RTN(8), "gptq": lambda: GPTQ(8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--bases", nargs="+", default=["rtn", "gptq"], choices=list(BASES))
    ap.add_argument("--choices", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--n-probes", type=int, default=64)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir, device=args.device)
    test = load_test_data(cfg, "test")
    train = load_test_data(cfg, "train")
    g = t.Generator().manual_seed(0)
    calib = train[t.randperm(len(train), generator=g)[: args.n_calib]]

    rows = []
    for base_name in args.bases:
        for budget in args.budgets:
            q = HAWQv2(budget, BASES[base_name](), choices=tuple(args.choices), n_probes=args.n_probes)
            qm = q.quantize(model, calib)
            r = evaluate(qm, test, device=args.device)
            actual = count_bytes(qm, q.allocation)
            row = dict(q.spec.as_row(), actual_bytes=actual, loss=r["loss"],
                       loss_per_pos=[round(x, 4) for x in r["loss_per_pos"]],
                       allocation=dict(sorted(q.allocation.items())),
                       traces={k: round(v, 2) for k, v in sorted(q.traces.items())})
            rows.append(row)
            print(f"base={base_name} budget={budget}  actual={actual}  loss={r['loss']:.4f}")
            bits_used = sorted(set(q.allocation.values()))
            print("  bits histogram:", {b: sum(1 for v in q.allocation.values() if v == b) for b in bits_used})

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            keys = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()})
    print("\ntraces (first run):")
    for k, v in rows[0]["traces"].items():
        print(f"  {k:26s} {v:12.2f}")


if __name__ == "__main__":
    main()
