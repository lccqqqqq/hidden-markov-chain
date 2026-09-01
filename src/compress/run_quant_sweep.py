"""
Comparative quantization sweep on one saved run.

  uv run python src/compress/run_quant_sweep.py models/cylinder_graph/<run_dir> \
      --methods rtn --bits 16 8 6 5 4 3 2 --out results/compression/<name>.csv

Writes one CSV row per (method, bits, granularity, symmetric) with test loss (nats/token),
per-position losses, parameter count and computed bytes, and prints a markdown table.
Reference rows: fp32 model, Bayes-optimal ctx floor, entropy rate, and the HMM's own dof.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path when run as a script
from compress import RTN, count_bytes, count_params
from compress.base import Quantizer
from compress.gptq import GPTQ
from evaluate import bayes_optimal, entropy_rate, evaluate, generator_dof, load_run, load_test_data
from utils import create_process_from_dict

METHODS: dict[str, type[Quantizer]] = {"rtn": RTN, "gptq": GPTQ}
# TODO: register AWQ, STEQAT, LSQ, HAWQv2 here once implemented (compress/*.py).
# Experimental methods register under a distinct prefix, e.g. "exp/<name>".


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--methods", nargs="+", default=["rtn"], choices=list(METHODS))
    ap.add_argument("--bits", nargs="+", type=int, default=[16, 8, 6, 5, 4, 3, 2])
    ap.add_argument("--granularity", nargs="+", default=["per_channel", "per_tensor"])
    ap.add_argument("--symmetric", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n-calib", type=int, default=128, help="train sequences for data-dependent methods")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir, args.ckpt, args.device)
    test = load_test_data(cfg, "test")
    train = load_test_data(cfg, "train")
    g = t.Generator().manual_seed(0)
    calib = train[t.randperm(len(train), generator=g)[: args.n_calib]]
    process = create_process_from_dict(cfg["data_generator"]["process"])

    n_params = count_params(model)
    bayes = bayes_optimal(process, test)
    h = entropy_rate(process)
    base = evaluate(model, test, device=args.device)

    rows = [dict(method="fp32", bits=32, granularity="-", symmetric="-", params=n_params,
                 bytes=count_bytes(model, None), loss=base["loss"], loss_per_pos=base["loss_per_pos"])]
    for m in args.methods:
        cls = METHODS[m]
        for bits in args.bits:
            for gran in args.granularity:
                for sym in args.symmetric:
                    if sym and bits < 2:
                        continue  # degenerate grid (see Quantizer.grid)
                    q = cls(bits, gran, bool(sym))
                    qm = q.quantize(model, calib)
                    r = evaluate(qm, test, device=args.device)
                    row = dict(q.spec.as_row(), params=n_params, bytes=count_bytes(qm, bits, gran),
                               loss=r["loss"], loss_per_pos=r["loss_per_pos"])
                    rows.append(row)
                    print(f"{q.name:8s} b={bits:2d} {gran:11s} sym={sym}  loss={r['loss']:.4f}  bytes={row['bytes']}")

    refs = dict(bayes_optimal=bayes["loss"], entropy_rate=h, hmm_dof=generator_dof(process),
                hmm_bytes_fp32=4 * generator_dof(process), params=n_params, bytes_fp32=rows[0]["bytes"])

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        keys = ["method", "bits", "granularity", "symmetric", "params", "bytes", "loss", "loss_per_pos"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({**r, "loss_per_pos": json.dumps([round(x, 4) for x in r["loss_per_pos"]])})
        with open(out.with_suffix(".refs.json"), "w") as f:
            json.dump(refs, f, indent=1)

    # markdown table
    print(f"\nrun: {args.run_dir}   params={n_params}   fp32 bytes={refs['bytes_fp32']}")
    print(f"Bayes-optimal (ctx {test.shape[1]-1}) = {bayes['loss']:.4f} nats   h_X = {h:.4f} nats   "
          f"HMM dof = {refs['hmm_dof']} ({refs['hmm_bytes_fp32']} bytes fp32)")
    print("\n| method | bits | granularity | sym | bytes | ×HMM bytes | loss (nats) | Δ vs fp32 | Δ vs Bayes | last-pos loss |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['method']} | {r['bits']} | {r['granularity']} | {r['symmetric']} | {r['bytes']} | "
              f"{r['bytes']/refs['hmm_bytes_fp32']:.0f} | {r['loss']:.4f} | {r['loss']-base['loss']:+.4f} | "
              f"{r['loss']-bayes['loss']:+.4f} | {r['loss_per_pos'][-1]:.4f} |")


if __name__ == "__main__":
    main()
