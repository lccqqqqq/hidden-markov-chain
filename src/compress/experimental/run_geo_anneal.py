"""
R1.1 / R1.2 runner.

  uv run python -u src/compress/experimental/run_geo_anneal.py models/cylinder_graph/<run> \
      --stage r11 r12 --threads 10 --out results/compression/geometry/

r11: for every hook gate (qk, head, attn, mlp, ngrp): quench (from E1 JSON if present, else
     measured), linear response (CG), anneal {50,200,800}+100 hold, quench+recover
     {150,300,900} steps. -> r1_adiabatic.json
r12: the 16 gates with largest quench cost: down-and-up hysteresis (200+100 each leg),
     3 seeds. -> r1_hysteresis.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_anneal import Annealer
from compress.experimental.geo_probes import Gate
from compress.experimental.teacher import calib_split, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", nargs="+", default=["r11", "r12"])
    ap.add_argument("--anneal", nargs="+", type=int, default=[50, 200, 800])
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--kinds", nargs="+", default=["qk", "head", "attn", "mlp", "ngrp"])
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
    an = Annealer(model, train, proc, calib, p_calib, test, p_test)
    gates = [g for g in an.gm.all_gates() if g.kind in args.kinds]
    kl0 = an.base_c["kl"]
    print(f"base calib KL {kl0:.5f}  test KL {an.base_t['kl']:.5f}  gates {len(gates)}", flush=True)

    quench = {}
    e1 = out / "e1_profiles.json"
    if e1.exists():
        pr = json.load(open(e1))
        quench = {k: v["kl"][0] - pr["base_calib"]["kl"] for k, v in pr["profiles"].items()}

    if "r11" in args.stage:
        res = {"base_calib_kl": kl0, "base_test_kl": an.base_t["kl"], "gates": {}}
        t0 = time.time()
        for k, g in enumerate(gates):
            rec = {"quench": quench.get(g.label)}
            if rec["quench"] is None:
                rec["quench"] = an.gm.measure([(g, 0.0)], calib, p_calib)["kl"] - kl0
            rec["linear"] = an.linear_response(g)
            rec["anneal"] = {str(T): an.run(g, T, args.hold) for T in args.anneal}
            rec["quench_recover"] = {str(T + args.hold): an.run(g, 0, T + args.hold) for T in args.anneal}
            res["gates"][g.label] = rec
            ad = {T: 1e3 * (r["end_calib_kl"] - kl0) for T, r in rec["anneal"].items()}
            qr = {T: 1e3 * (r["end_calib_kl"] - kl0) for T, r in rec["quench_recover"].items()}
            print(f"[{k+1:3d}/{len(gates)}] {g.label:12s} quench={1e3*rec['quench']:7.2f}  "
                  f"lin={1e3*rec['linear']['linear_response']:7.2f} (quad {1e3*rec['linear']['quench_quad']:6.2f}, "
                  f"cg_res {rec['linear']['cg_residual']:.1e})  anneal={ {a: round(b,2) for a,b in ad.items()} }  "
                  f"quench+rec={ {a: round(b,2) for a,b in qr.items()} }  [{time.time()-t0:.0f}s]", flush=True)
            with open(out / "r1_adiabatic.json", "w") as f:
                json.dump(res, f)

    if "r12" in args.stage:
        top = sorted(gates, key=lambda g: -quench.get(g.label, 0.0))[:16]
        res = {"base_calib_kl": kl0, "base_test_kl_per_pos": an.base_t["kl_per_pos"], "gates": {}}
        t0 = time.time()
        for k, g in enumerate(top):
            runs = [an.run(g, 200, args.hold, seed=s, return_up=True) for s in range(3)]
            res["gates"][g.label] = dict(quench=quench.get(g.label), runs=runs)
            print(f"[{k+1:2d}/16] {g.label:12s} quench={1e3*quench.get(g.label,0):7.2f}  "
                  f"down={ [round(1e3*(r['down_calib_kl']-kl0),2) for r in runs] }  "
                  f"return={ [round(1e3*(r['end_calib_kl']-kl0),2) for r in runs] }  "
                  f"disp={ [round(r['displacement'],2) for r in runs] }  [{time.time()-t0:.0f}s]", flush=True)
            with open(out / "r1_hysteresis.json", "w") as f:
                json.dump(res, f)


if __name__ == "__main__":
    main()
