"""
§4.3 runner — finite-temperature correlators of coarse observables and the compression they enable.

  uv run python -u src/compress/experimental/run_geo_correlators.py models/cylinder_graph/<run> \
      --stage traj delete tgptq --dynamics adam sgd sgld --threads 8 --out results/compression/geometry

Stages
  traj    : trajectories (adam lr 1e-3 / sgd lr 1e-2 / sgld eps 3e-7 with nbeta = 16n/log n, gamma 100),
            correlator analysis, comparisons with R3.3 (chi), R3.1 (free fraction, field cost),
            E1 (quench) and R1 (relaxed).                      -> r4_corr_<dyn>.json, r4_summary.json
  delete  : deletion order by thermal variance (adam) vs quench-cost order vs random;
            hard delete + 200 soft recovery steps at matched removed-parameter targets;
            compared with R3.2's global-field frontier.          -> r4_delete.csv
  tgptq   : GPTQ with the thermal activation Hessian <2XX^T> (adam trajectory, 50 states)
            vs H(w*) without propagation vs reference propagated GPTQ, bits 4/3/2.  -> r4_tgptq.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.experimental.geo_correlators import (FixedHGPTQ, activation_hessians, analyze,
                                                   thermal_hessians, thermal_trajectory)
from compress.experimental.geo_field import FieldModel
from compress.experimental.geo_probes import Gate, GateModel
from compress.experimental.geo_recover import recover
from compress.experimental.geo_alloc import ExtGPTQ
from compress.experimental.teacher import calib_split, evaluate_teacher, teacher_probs
from compress.gptq import GPTQ
from evaluate import load_run, load_test_data
from utils import create_process_from_dict


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 3 else float("nan")


def load_field_sweeps(out: Path) -> dict:
    """Merge r3_field.json from the local run and the hydra shards."""
    merged = {}
    cands = [out / "r3_field.json"]
    for base in (out / "hydra", out):
        cands += [base / d / "r3_field.json" for d in ("hydra_r3main", "hydra_r3ngrp", "hydra_r3qk")]
    for p in cands:
        if p.exists():
            for k, v in json.load(open(p)).items():
                if isinstance(v, dict) and "m" in v:
                    merged.setdefault(k, v)
    return merged


def gate_from_label(label: str, gm: GateModel) -> Gate:
    kind, rest = label.split(".", 1)
    parts = rest.split(".")
    layer = int(parts[0][1:]); idx = int(parts[1]) if len(parts) > 1 else -1
    return Gate(kind, layer, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", nargs="+", default=["traj", "delete", "tgptq"])
    ap.add_argument("--dynamics", nargs="+", default=["adam", "sgd", "sgld"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--recover-steps", type=int, default=200)
    ap.add_argument("--n-states", type=int, default=50)
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    calib = calib_split(train, args.n_calib, seed=0)
    p_calib, p_test = teacher_probs(proc, calib), teacher_probs(proc, test)
    gm = GateModel(model, calib)
    fm = FieldModel(model, gm, proc, calib[:512])
    labels = [g.label for g in fm.gates]
    base_t = evaluate_teacher(model, test, p_test)
    n = len(train); nbeta = 16 * n / math.log(n)
    LR = {"adam": 1e-3, "sgd": 1e-2, "sgld": 3e-7}
    print(f"fp32 test nll {base_t['nll']:.4f} kl {base_t['kl']:.5f}; {len(labels)} observables", flush=True)

    # reference data for comparisons
    e1 = json.load(open(out / "e1_profiles.json")) if (out / "e1_profiles.json").exists() else None
    quench = {k: v["kl"][0] - e1["base_calib"]["kl"] for k, v in e1["profiles"].items()} if e1 else {}
    sweeps = load_field_sweeps(out)
    free_frac = {k: 1.0 - v["m"][-1] for k, v in sweeps.items()}                 # at the strongest field
    field_cost = {k: v["dkl_test"][-1] for k, v in sweeps.items()}
    chi = None
    for cross_p in (out / "hydra" / "hydra_r3main" / "r3_cross.json", out / "hydra_r3main" / "r3_cross.json"):
        if cross_p.exists():
            chi = json.load(open(cross_p)); break

    if "traj" in args.stage:
        summary = {}
        for dyn in args.dynamics:
            t0 = time.time()
            steps = args.steps * (2 if dyn == "sgld" else 1)
            tr = thermal_trajectory(model, fm, train, proc, dyn, steps, LR[dyn], record_every=args.record_every,
                                    nbeta=nbeta, calib=calib, p_calib=p_calib)
            an = analyze(tr)
            an.update(dynamics=dyn, lr=LR[dyn], steps=steps, dist=tr["dist"], kl_steps=tr["kl_steps"], kl=tr["kl"],
                      batch_loss=tr["batch_loss"], diverged=tr["diverged"])
            var = dict(zip(labels, an["var"])); lc = dict(zip(labels, an["loss_corr"])); tau = dict(zip(labels, an["tau_records"]))
            comp = {}
            common = [k for k in labels if k in quench]
            comp["spearman_var_vs_quench"] = spearman([var[k] for k in common], [quench[k] for k in common])
            common_f = [k for k in labels if k in free_frac]
            comp["spearman_var_vs_freefrac"] = spearman([var[k] for k in common_f], [free_frac[k] for k in common_f])
            comp["spearman_losscorr_vs_fieldcost"] = spearman([lc[k] for k in common_f], [field_cost[k] for k in common_f])
            comp["spearman_abs_losscorr_vs_quench"] = spearman([abs(lc[k]) for k in common], [quench[k] for k in common])
            if chi is not None:
                corr = np.array(an["corr"]); li = {k: i for i, k in enumerate(labels)}
                a, b = [], []
                for r in chi["rows"]:
                    for c in chi["cols"]:
                        if r != c and r in li and c in li:
                            a.append(corr[li[r], li[c]]); b.append(chi["chi"][r][c])
                comp["spearman_corr_vs_chi"] = spearman(a, b)
                comp["sign_agreement_corr_vs_minus_chi"] = float(np.mean(np.sign(a) == -np.sign(b)))   # FDT: dΩ/dh = -β cov
                big = [i for i in range(len(b)) if abs(b[i]) > 0.2]
                comp["sign_agreement_large_chi"] = float(np.mean([np.sign(a[i]) == -np.sign(b[i]) for i in big])) if big else float("nan")
                comp["n_large_chi"] = len(big)
            an["comparisons"] = comp
            with open(out / f"r4_corr_{dyn}.json", "w") as f:
                json.dump(an, f)
            top = sorted(labels, key=lambda k: -var[k])[:6]; bot = sorted(labels, key=lambda k: var[k])[:4]
            print(f"[{dyn}] steps={steps} lr={LR[dyn]:g} diverged={tr['diverged']} |w-w*|_end={tr['dist'][-1]:.2f} "
                  f"KL_end={tr['kl'][-1]*1e3:.2f} mnat  samples={an['n_samples']}  [{time.time()-t0:.0f}s]", flush=True)
            print(f"   loosest (var): {[(k, f'{var[k]:.2e}') for k in top]}\n   tightest: {[(k, f'{var[k]:.2e}') for k in bot]}", flush=True)
            print(f"   tau (records) median {np.nanmedian(list(tau.values())):.1f}, max {np.nanmax(list(tau.values())):.1f}", flush=True)
            print("   comparisons: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in comp.items()), flush=True)
            summary[dyn] = comp
        with open(out / "r4_summary.json", "w") as f:
            json.dump(summary, f, indent=1)

    if "delete" in args.stage:
        from compress.experimental.run_geo_globalfield import removed_params
        an = json.load(open(out / "r4_corr_adam.json"))
        var = dict(zip(an["labels"], an["var"]))
        pd = json.load(open(out / "r3_globalfield.json")) if (out / "r3_globalfield.json").exists() else None
        targets = [d["removed_params"] for d in pd["deletions"] if d["removed_params"] > 0] if pd else [40000, 95000, 128000, 140000, 165000]
        rng = np.random.default_rng(0)
        fine = [k for k in labels if k.split(".")[0] in ("qk", "head", "ngrp")]
        orders = {"thermal_var_desc": sorted(labels, key=lambda k: -var[k]),
                  "thermal_var_desc_fine": sorted(fine, key=lambda k: -var[k]),
                  "quench_asc": sorted([k for k in labels if k in quench], key=lambda k: quench[k]),
                  "random": list(rng.permutation(labels))}
        rows = []
        total = sum(p.numel() for p in model.parameters())
        for oname, order in orders.items():
            for target in targets:
                dead, removed = [], 0
                for lab in order:
                    g = gate_from_label(lab, gm)
                    cand = dead + [g]
                    removed = removed_params(cand, model.cfg)
                    dead = cand
                    if removed >= target:
                        break
                import copy
                hard = copy.deepcopy(model)
                gmh = GateModel(hard, calib); gmh.groups = gm.groups
                with gmh.applied([(g, 0.0) for g in dead]):
                    r0 = evaluate_teacher(hard, test, p_test)
                    recover(hard, train, proc, args.recover_steps, soft_targets=True)
                    r1 = evaluate_teacher(hard, test, p_test)
                ref = None
                if pd:
                    ref = min(pd["deletions"], key=lambda d: abs(d["removed_params"] - removed))
                rows.append(dict(order=oname, target=target, removed_params=removed, params_left=total - removed,
                                 n_dead=len(dead), deleted_nll=r0["nll"], recovered_nll=r1["nll"],
                                 d_mnat=1e3 * (r1["nll"] - base_t["nll"]),
                                 r32_ref_params_left=ref["params_left"] if ref else None,
                                 r32_ref_d_mnat=1e3 * (ref["test_nll_recovered"] - base_t["nll"]) if ref else None,
                                 dead=";".join(g.label for g in dead)))
                print(f"[delete] {oname:17s} removed={removed:6d} left={total-removed:6d} n={len(dead):2d} "
                      f"deleted={r0['nll']:.4f} -> recovered {r1['nll']:.4f} (d={rows[-1]['d_mnat']:+.1f})"
                      + (f"  | R3.2 @ {ref['params_left']}: d={rows[-1]['r32_ref_d_mnat']:+.1f}" if ref else ""), flush=True)
                with open(out / "r4_delete.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    if "tgptq" in args.stage:
        gcal = calib[:128]
        H0 = activation_hessians(model, gcal[:, :-1])
        HT = thermal_hessians(model, fm, train, proc, gcal[:, :-1], "adam", args.steps, LR["adam"], args.n_states)
        rows = []
        for bits in (4, 3, 2):
            for name, q in (("GPTQ-ref(propagated)", GPTQ(bits)), ("GPTQ-H(w*)-noprop", FixedHGPTQ(bits, H0)),
                            (f"ThermalGPTQ-{args.n_states}states", FixedHGPTQ(bits, HT))):
                qm = q.quantize(model, gcal)
                r = evaluate_teacher(qm, test, p_test)
                rows.append(dict(method=name, bits=bits, test_nll=r["nll"], test_kl=r["kl"], d_mnat=1e3 * (r["nll"] - base_t["nll"])))
                print(f"[tgptq] {name:26s} b={bits} nll={r['nll']:.4f} (d={rows[-1]['d_mnat']:+.1f})", flush=True)
        with open(out / "r4_tgptq.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
