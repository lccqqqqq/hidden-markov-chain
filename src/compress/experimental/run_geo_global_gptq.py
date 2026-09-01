"""
E7 runner: reference GPTQ vs GlobalGPTQ at bits {4,3,2} x act_order {F,T} x damp_newton sweep.

  uv run python -u src/compress/experimental/run_geo_global_gptq.py \
      models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN \
      --bits 4 3 2 --damp-newton 1e-3 1e-2 1e-1 --out results/compression/geometry/e7_global_gptq.csv
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
from compress.experimental.geo_global_gptq import GlobalGPTQ
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from compress.gptq import GPTQ
from evaluate import load_run, load_test_data
from utils import create_process_from_dict

COLS = ["method", "bits", "act_order", "damp_newton", "cg_iters", "n_newton_calib", "newton_skips",
        "test_nll", "test_kl", "wall_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    ap.add_argument("--act-order", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--damp-newton", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--cg-iters", type=int, default=20)
    ap.add_argument("--n-newton-calib", type=int, default=1024)
    ap.add_argument("--n-gptq-calib", type=int, default=128)
    ap.add_argument("--out", default="results/compression/geometry/e7_global_gptq.csv")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--log-json", default=None, help="optional per-run Newton logs")
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    p_test = teacher_probs(proc, test)
    pool = calib_split(train, args.n_gptq_calib + args.n_newton_calib, seed=0)
    gptq_calib, newton_calib = pool[: args.n_gptq_calib], pool[args.n_gptq_calib:]
    p_newton = teacher_probs(proc, newton_calib)
    base = evaluate_teacher(model, test, p_test)
    print(f"fp32 test nll={base['nll']:.4f} kl={base['kl']:.5f}", flush=True)

    rows, logs = [], {}
    for bits in args.bits:
        for ao in args.act_order:
            t0 = time.time()
            qm = GPTQ(bits, n_calib=args.n_gptq_calib, act_order=bool(ao)).quantize(model, gptq_calib)
            r = evaluate_teacher(qm, test, p_test)
            rows.append(dict(method="GPTQ", bits=bits, act_order=bool(ao), damp_newton="", cg_iters="",
                             n_newton_calib="", newton_skips="", test_nll=r["nll"], test_kl=r["kl"],
                             wall_seconds=round(time.time() - t0, 1)))
            print(f"GPTQ        b={bits} ao={ao}            nll={r['nll']:.4f} (d={1e3*(r['nll']-base['nll']):+.1f}) "
                  f"kl={r['kl']:.5f} [{time.time()-t0:.0f}s]", flush=True)
            for dn in args.damp_newton:
                t0 = time.time()
                q = GlobalGPTQ(bits, n_calib=args.n_gptq_calib, act_order=bool(ao), damp_newton=dn,
                               cg_iters=args.cg_iters, n_newton_calib=args.n_newton_calib)
                q.newton_calib, q.newton_probs = newton_calib, p_newton
                qm = q.quantize(model, gptq_calib)
                r = evaluate_teacher(qm, test, p_test)
                rows.append(dict(method="GlobalGPTQ", bits=bits, act_order=bool(ao), damp_newton=dn,
                                 cg_iters=args.cg_iters, n_newton_calib=args.n_newton_calib,
                                 newton_skips=q.newton_skips, test_nll=r["nll"], test_kl=r["kl"],
                                 wall_seconds=round(time.time() - t0, 1)))
                logs[f"b{bits}_ao{ao}_dn{dn:g}"] = q.newton_log
                print(f"GlobalGPTQ  b={bits} ao={ao} dn={dn:<7g} nll={r['nll']:.4f} (d={1e3*(r['nll']-base['nll']):+.1f}) "
                      f"kl={r['kl']:.5f} skips={q.newton_skips} [{time.time()-t0:.0f}s]", flush=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
            if args.log_json:
                with open(args.log_json, "w") as f:
                    json.dump(logs, f)


if __name__ == "__main__":
    main()
