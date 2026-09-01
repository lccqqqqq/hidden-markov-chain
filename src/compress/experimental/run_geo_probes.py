"""
Run E1 (profiles + controls) and E2 (pairwise grids).

  uv run python src/compress/experimental/run_geo_probes.py models/cylinder_graph/<run> \
      --stage e1 e2 --out results/compression/geometry/

Outputs (JSON):
  e1_profiles.json   {gate_label: {alphas, nll, kl, g_dot_d, dHd, loss0_nll, loss0_kl}}
  e1_controls.json   uniform-attention / no-QK / context-shuffle controls on the TEST split
  e2_pairs.json      {pair_label: {alphas, kl grid (5x5), cross_hessian, unary_i, unary_j}}
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_probes import Gate, GateModel, QuadraticModel
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", nargs="+", default=["e1", "e2"])
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=20)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test = load_test_data(cfg, "test")
    train = load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)

    gm = GateModel(model, calib)
    base_c = evaluate_teacher(model, calib, p_calib)
    base_t = evaluate_teacher(model, test, p_test)
    print(f"fp32  calib: nll {base_c['nll']:.4f} kl {base_c['kl']:.5f} | test: nll {base_t['nll']:.4f} kl {base_t['kl']:.5f}")

    quad = QuadraticModel(model, calib, p_calib)
    print(f"quadratic model built on calib (KL at w* = {quad.loss0:.5f}); |grad| = "
          f"{sum(g.detach().pow(2).sum() for g in quad.grads).sqrt():.4e}")

    alphas = [round(0.1 * i, 1) for i in range(21)]
    gates = gm.all_gates()
    dirs = {g.label: gm.direction(g) for g in gates}
    Hd = {}

    if "e1" in args.stage:
        t0 = time.time()
        res = {"base_calib": base_c, "base_test": base_t, "alphas": alphas, "profiles": {}}
        for k, g in enumerate(gates):
            d = dirs[g.label]
            Hd[g.label] = quad.hvp(d)
            gd, dHd = quad.gdot(d), quad.dot(Hd[g.label], d)
            nll, kl = [], []
            for a in alphas:
                r = gm.measure([(g, a)], calib, p_calib) if a != 1.0 else base_c
                nll.append(r["nll"]); kl.append(r["kl"])
            res["profiles"][g.label] = dict(nll=nll, kl=kl, g_dot_d=gd, dHd=dHd,
                                             d_norm=float(sum(v.pow(2).sum() for v in d.values()).sqrt()))
            kl0 = base_c["kl"]
            print(f"[{k+1:3d}/{len(gates)}] {g.label:12s} dHd={dHd:9.3e}  "
                  f"dKL(a=0)={kl[0]-kl0:+.4f} dKL(a=.5)={kl[5]-kl0:+.4f} dKL(a=1.5)={kl[15]-kl0:+.4f} "
                  f"dKL(a=2)={kl[20]-kl0:+.4f}   quad(a=0)={0.5*dHd - gd:+.4f}  [{time.time()-t0:.0f}s]")
        with open(out / "e1_profiles.json", "w") as f:
            json.dump(res, f)

        # ---- controls on the TEST split ---------------------------------------------------
        ctrl = {"base_test": base_t}
        allqk = [(Gate("qk", l, h), 0.0) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
        ctrl["uniform_attention_all"] = gm.measure(allqk, test, p_test)
        noqk = ([(Gate("W_Q", l, h), 0.0) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
                + [(Gate("W_K", l, h), 0.0) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)])
        ctrl["zero_WQ_WK_all"] = gm.measure(noqk, test, p_test)
        for l in range(model.cfg.n_layers):
            ctrl[f"uniform_attention_L{l}"] = gm.measure(
                [(Gate("qk", l, h), 0.0) for h in range(model.cfg.n_heads)], test, p_test)
        # context shuffle: last position only, target fixed
        seq = test.shape[1] - 1
        for variant, keep_last in (("shuffle_all_context", False), ("shuffle_all_but_last", True)):
            vals_nll, vals_kl = [], []
            for seed in range(3):
                g = t.Generator().manual_seed(seed)
                sh = test.clone()
                n_perm = seq - 1 if keep_last else seq
                perm = t.argsort(t.rand(len(test), n_perm, generator=g), dim=1)
                sh[:, :n_perm] = t.gather(test[:, :n_perm], 1, perm)
                r = evaluate_teacher(model, sh, p_test)      # teacher probs from ORIGINAL order
                vals_nll.append(r["nll_per_pos"][-1]); vals_kl.append(r["kl_per_pos"][-1])
            ctrl[variant] = dict(last_pos_nll=vals_nll, last_pos_kl=vals_kl,
                                 base_last_pos_nll=base_t["nll_per_pos"][-1], base_last_pos_kl=base_t["kl_per_pos"][-1])
        with open(out / "e1_controls.json", "w") as f:
            json.dump(ctrl, f, indent=1)
        print("controls:", json.dumps({k: (v.get("nll"), v.get("kl")) if "nll" in v else v for k, v in ctrl.items()}, indent=1))

    if "e2" in args.stage:
        t0 = time.time()
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        heads = [g for g in gates if g.kind == "head"]
        subl = [g for g in gates if g.kind in ("attn", "mlp")]
        for g in heads + subl:
            if g.label not in Hd:
                Hd[g.label] = quad.hvp(dirs[g.label])
        unary = {}
        for g in heads + subl:
            unary[g.label] = [gm.measure([(g, a)], calib, p_calib)["kl"] if a != 1.0 else base_c["kl"] for a in grid]
        pairs = list(itertools.combinations(heads, 2)) + list(itertools.combinations(subl, 2))
        res = {"base_kl": base_c["kl"], "grid": grid, "unary": unary, "pairs": {}}
        for k, (gi, gj) in enumerate(pairs):
            M = [[None] * len(grid) for _ in grid]
            for a, ai in enumerate(grid):
                for b, bj in enumerate(grid):
                    if ai == 1.0:
                        M[a][b] = unary[gj.label][b]
                    elif bj == 1.0:
                        M[a][b] = unary[gi.label][a]
                    else:
                        M[a][b] = gm.measure([(gi, ai), (gj, bj)], calib, p_calib)["kl"]
            cross = quad.dot(Hd[gi.label], dirs[gj.label])
            res["pairs"][f"{gi.label}|{gj.label}"] = dict(kl=M, cross_hessian=cross,
                                                          dHd_i=quad.dot(Hd[gi.label], dirs[gi.label]),
                                                          dHd_j=quad.dot(Hd[gj.label], dirs[gj.label]))
            # non-additivity at (0,0)
            R00 = M[0][0] - M[0][4] - M[4][0] + base_c["kl"]
            print(f"[{k+1:3d}/{len(pairs)}] {gi.label:10s}|{gj.label:10s} R(0,0)={R00:+.4f} "
                  f"quad cross={cross:+.3e} [{time.time()-t0:.0f}s]")
            if (k + 1) % 20 == 0:
                with open(out / "e2_pairs.json", "w") as f:
                    json.dump(res, f)
        with open(out / "e2_pairs.json", "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
