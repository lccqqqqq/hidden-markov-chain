"""
R3.2 — global-field phase diagram (relaxation_program.md).

One field on all 72 components at once, group-lasso style:
    penalty(h) = h * sum_c m_c(w),   m_c = sqrt(Omega_c(w) / Omega_c(w*))
(Omega_c are the gauge-invariant component fields of geo_field.py; m_c is the component's
normalized norm, 1 at w*). h is swept upward (annealed: each h starts from the previous
state), then back down for hysteresis. At each upward h the components with m_c < thresh
are hard-deleted (gate alpha = 0 via GateModel) and the loss is measured before and after
200 recovery steps -> a loss-vs-parameters curve produced by the field, comparable with
E4/FWSVD/pruning rows.

  uv run python -u src/compress/experimental/run_geo_globalfield.py models/cylinder_graph/<run> --threads 10
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_field import FieldModel, act_hook_name, field_gates, param_field
from compress.experimental.geo_probes import Gate, GateModel
from compress.experimental.geo_recover import recover
from compress.experimental.teacher import calib_split, evaluate_teacher, kl_loss, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def params_of(gate: Gate, cfg) -> int:
    dm, dh, dmlp = cfg.d_model, cfg.d_head, cfg.d_mlp
    if gate.kind == "head":
        return 4 * dm * dh + 3 * dh
    if gate.kind == "qk":
        return 2 * dm * dh + 2 * dh
    if gate.kind == "ngrp":
        return 32 * (2 * dm + 1)
    if gate.kind == "attn":
        return cfg.n_heads * (4 * dm * dh + 3 * dh) + dm
    if gate.kind == "mlp":
        return dmlp * (2 * dm + 1) + dm
    raise ValueError(gate.kind)


def removed_params(dead: list[Gate], cfg) -> int:
    """Count parameters removed by a set of gates without double counting nested ones."""
    total = 0
    dead_set = {g.label for g in dead}
    for g in dead:
        if g.kind == "attn" or g.kind == "mlp":
            total += params_of(g, cfg)
        elif g.kind == "head":
            if f"attn.L{g.layer}" not in dead_set:
                total += params_of(g, cfg)
        elif g.kind == "qk":
            if f"attn.L{g.layer}" not in dead_set and f"head.L{g.layer}.{g.idx}" not in dead_set:
                total += params_of(g, cfg)
        elif g.kind == "ngrp":
            if f"mlp.L{g.layer}" not in dead_set:
                total += params_of(g, cfg)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--h-mnat", nargs="+", type=float, default=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--recover-steps", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out)

    model0, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
    gm = GateModel(model0, calib)
    fm = FieldModel(model0, gm, proc, calib[:512])
    gates = fm.gates
    om0 = fm.omegas(model0)
    base_c, base_t = evaluate_teacher(model0, calib, p_calib), evaluate_teacher(model0, test, p_test)
    total_params = sum(p.numel() for p in model0.parameters())
    act_gates = [g for g in gates if g.kind in ("attn", "mlp")]
    par_gates = [g for g in gates if g.kind not in ("attn", "mlp")]
    print(f"base calib KL {base_c['kl']:.5f} test KL {base_t['kl']:.5f}; {len(gates)} components", flush=True)

    def penalty(model, store):
        tot = 0.0
        for g in par_gates:
            tot = tot + t.sqrt(param_field(model, g, gm.groups) / om0[g.label] + 1e-12)
        for g in act_gates:
            tot = tot + t.sqrt(store[act_hook_name(g)].pow(2).sum(-1).mean() / om0[g.label] + 1e-12)
        return tot

    def train_at(model, h, steps, seed):
        opt = t.optim.Adam(model.parameters(), lr=args.lr)
        g = t.Generator().manual_seed(seed)
        model.train()
        for s in range(steps):
            idx = t.randint(0, len(train), (128,), generator=g)
            batch = train[idx]
            store = {}
            for ag in act_gates:
                model.add_hook(act_hook_name(ag), lambda a, hook: store.__setitem__(hook.name, a))
            try:
                kl = kl_loss(model, batch, teacher_probs(proc, batch))
                loss = kl + h * penalty(model, store)
            finally:
                model.reset_hooks()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

    hs = [h * 1e-3 for h in args.h_mnat]
    path = hs + hs[::-1][1:]                     # up then down (hysteresis)
    model = copy.deepcopy(model0)
    res = dict(base_calib_kl=base_c["kl"], base_test_kl=base_t["kl"], base_test_nll=base_t["nll"],
               total_params=total_params, h_mnat=[h * 1e3 for h in path], leg=["up"] * len(hs) + ["down"] * (len(hs) - 1),
               m=[], calib_kl=[], test_kl=[], test_nll=[], deletions=[])
    t0 = time.time()
    for k, h in enumerate(path):
        train_at(model, h, args.steps, seed=k)
        om = fm.omegas(model)
        m = {g.label: math.sqrt(max(om[g.label], 0.0) / om0[g.label]) for g in gates}
        rc, rt = evaluate_teacher(model, calib, p_calib), evaluate_teacher(model, test, p_test)
        res["m"].append(m); res["calib_kl"].append(rc["kl"]); res["test_kl"].append(rt["kl"]); res["test_nll"].append(rt["nll"])
        dead = [g for g in gates if m[g.label] < args.thresh]
        n_dead = {kd: sum(1 for g in dead if g.kind == kd) for kd in ("qk", "head", "ngrp", "attn", "mlp")}
        line = (f"[{k+1:2d}/{len(path)}] {res['leg'][k]:4s} h={h*1e3:6.1f} mnat  dKL test={1e3*(rt['kl']-base_t['kl']):+7.2f}  "
                f"dead={n_dead}  removed_params={removed_params(dead, model.cfg)}")
        if res["leg"][k] == "up":
            # hard delete + recovery from the *field-trained* state
            hard = copy.deepcopy(model)
            gmh = GateModel(hard, calib); gmh.groups = gm.groups
            with gmh.applied([(g, 0.0) for g in dead]):
                r_del = evaluate_teacher(hard, test, p_test)
                recover(hard, train, proc, args.recover_steps, soft_targets=True)
                r_rec = evaluate_teacher(hard, test, p_test)
            n_rem = removed_params(dead, model.cfg)
            res["deletions"].append(dict(h_mnat=h * 1e3, dead=[g.label for g in dead], removed_params=n_rem,
                                         params_left=total_params - n_rem, test_nll_deleted=r_del["nll"],
                                         test_nll_recovered=r_rec["nll"], test_kl_recovered=r_rec["kl"]))
            line += f"  | hard-delete: nll {r_del['nll']:.4f} -> +{args.recover_steps} steps {r_rec['nll']:.4f} (d={1e3*(r_rec['nll']-base_t['nll']):+.1f}) params_left={total_params-n_rem}"
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
        with open(out / "r3_globalfield.json", "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
