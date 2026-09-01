"""
R3.1 / R3.3 runner.

  uv run python -u src/compress/experimental/run_geo_field.py models/cylinder_graph/<run> \
      --stage r31 r33 --steps 300 --threads 8 --out results/compression/geometry

Outputs: r3_field.json  (per component: h_values, targets_mnat, m, dkl_calib, dkl_test, displacement, omega0, jump)
         r3_cross.json  (chi matrix 24 x 72, drift-corrected, plus labels and the h used per row)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_field import (FieldModel, assert_gauge_invariant, field_gates, field_sweep,
                                             finetune_with_field, h_for_half)
from compress.experimental.geo_probes import GateModel
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", nargs="+", default=["r31", "r33"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kinds", nargs="+", default=["qk", "head", "attn", "mlp", "ngrp"])
    ap.add_argument("--only", nargs="+", default=None, help="gate labels to restrict to (smoke tests)")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--n-act", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/compression/geometry")
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
    gm = GateModel(model, calib)
    fm = FieldModel(model, gm, proc, calib[: args.n_act])
    kl0_c, kl0_t = evaluate_teacher(model, calib, p_calib)["kl"], evaluate_teacher(model, test, p_test)["kl"]
    dl, worst = assert_gauge_invariant(model, fm, calib)
    print(f"gauge check: max|dlogits|={dl:.2e}, worst field rel change={worst[0]:.2e} ({worst[1]}); "
          f"KL0 calib={kl0_c:.5f} test={kl0_t:.5f}", flush=True)
    gates = [g for g in field_gates(gm) if g.kind in args.kinds and (args.only is None or g.label in args.only)]
    omega0 = fm.omegas(model)

    if "r31" in args.stage:
        path = out / "r3_field.json"
        res = json.load(open(path)) if path.exists() else {}
        t0 = time.time()
        for k, g in enumerate(gates):
            if g.label in res:
                continue
            r = field_sweep(model, fm, g, train, calib, p_calib, test, p_test, kl0_c, kl0_t,
                            targets=args.targets, steps=args.steps, lr=args.lr)
            res[g.label] = r
            with open(path, "w") as f:
                json.dump(res, f)
            print(f"[{k+1:3d}/{len(gates)}] {g.label:12s} omega0={r['omega0']:.3e} {'JUMP' if r['jump'] else 'cont'} | "
                  f"m: " + " ".join(f"{m:.2f}" for m in r["m"]) + " | dKL(mnat): "
                  + " ".join(f"{1e3*d:.1f}" for d in r["dkl_calib"]) + f"  [{time.time()-t0:.0f}s]", flush=True)

    if "r33" in args.stage:
        sweeps = json.load(open(out / "r3_field.json")) if (out / "r3_field.json").exists() else {}
        rows = [g for g in gates if g.kind in ("head", "attn", "mlp")]
        labels = [g.label for g in field_gates(gm)]
        # drift baseline: h = 0 fine-tune
        base_model, _ = finetune_with_field(model, fm, rows[0], 0.0, train, args.steps, args.lr)
        om_base = fm.omegas(base_model)
        drift = {lab: (om_base[lab] - omega0[lab]) / max(abs(omega0[lab]), 1e-12) for lab in labels}
        chi, h_used = {}, {}
        t0 = time.time()
        for k, g in enumerate(rows):
            h = h_for_half(sweeps[g.label]) if g.label in sweeps else 10e-3 / max(omega0[g.label], 1e-12)
            m_h, _ = finetune_with_field(model, fm, g, h, train, args.steps, args.lr)
            om = fm.omegas(m_h)
            chi[g.label] = {lab: (om[lab] - omega0[lab]) / max(abs(omega0[lab]), 1e-12) - drift[lab] for lab in labels}
            h_used[g.label] = h
            own = chi[g.label][g.label]
            others = sorted(((v, lab) for lab, v in chi[g.label].items() if lab != g.label), key=lambda x: x[0])
            print(f"[{k+1:2d}/{len(rows)}] {g.label:10s} h={h:.2e} own m-1={own:+.2f} | most neg: "
                  + ", ".join(f"{lab}:{v:+.2f}" for v, lab in others[:3]) + " | most pos: "
                  + ", ".join(f"{lab}:{v:+.2f}" for v, lab in others[-3:]) + f"  [{time.time()-t0:.0f}s]", flush=True)
            with open(out / "r3_cross.json", "w") as f:
                json.dump(dict(rows=[r.label for r in rows], cols=labels, chi=chi, h_used=h_used, drift=drift,
                               steps=args.steps), f)
        pairs = []
        rowset = {r.label for r in rows}
        for c, d in chi.items():
            for c2, v in d.items():
                if c2 in rowset and c2 != c:
                    pairs.append((v, c, c2))
        pairs.sort()
        print("\nmost compensating (negative chi) head/sublayer pairs:")
        for v, c, c2 in pairs[:10]:
            print(f"  squeeze {c:10s} -> {c2:10s} {v:+.3f}")
        print("most cooperating (positive chi):")
        for v, c, c2 in pairs[-10:]:
            print(f"  squeeze {c:10s} -> {c2:10s} {v:+.3f}")


if __name__ == "__main__":
    main()
