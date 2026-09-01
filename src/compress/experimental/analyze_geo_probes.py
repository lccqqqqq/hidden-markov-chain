"""
Summarize E1 / E2 outputs.

  uv run python src/compress/experimental/analyze_geo_probes.py results/compression/geometry/

For each gate profile: measured dKL(alpha) vs quadratic prediction, the radius where the
quadratic model breaks (|measured - quad| > max(0.5 mnat, 25% of measured)), a power-law fit
dKL ~ c |alpha-1|^p on the shrink side (alpha < 1), and a shape class:
  flat     : dKL(alpha=0) < 1 mnat                       (component removable for free)
  quad     : quadratic model within tolerance over the full range
  cliff    : flat near alpha=1 (quad radius >= 0.3) but dKL(0) > 10 mnat  -> plateau + cliff
  power    : exponent p fitted on [0.5,0.9] in (2.5, 6) with R^2 > 0.95   -> higher-order
  other
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def classify(alphas, dkl, gd, dHd):
    a = np.array(alphas); d = np.array(dkl)
    quad = (a - 1) * gd + 0.5 * (a - 1) ** 2 * dHd
    err = np.abs(d - quad)
    tol = np.maximum(5e-4, 0.25 * np.abs(d))
    ok = err <= tol
    # radius of validity on each side
    left = [1 - a[i] for i in range(len(a)) if a[i] < 1 and not ok[i]]
    right = [a[i] - 1 for i in range(len(a)) if a[i] > 1 and not ok[i]]
    r_left = min(left) if left else 1.0
    r_right = min(right) if right else 1.0
    # power-law fit on shrink side alpha in [0.5, 0.9]
    m = (a >= 0.5) & (a <= 0.9) & (d > 1e-6)
    p = r2 = float("nan")
    if m.sum() >= 3:
        x, y = np.log(1 - a[m]), np.log(d[m])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        p = coef[0]
        yhat = A @ coef
        r2 = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    d0 = d[0]
    if d0 < 1e-3:
        cls = "flat"
    elif ok.all():
        cls = "quad"
    elif r_left >= 0.3 and d0 > 1e-2:
        cls = "cliff"
    elif 2.5 < p < 6 and r2 > 0.95:
        cls = "power"
    else:
        cls = "other"
    return dict(cls=cls, d0=d0, d_half=d[5], d_15=d[15], d_2=d[20], quad0=quad[0], r_left=r_left,
                r_right=r_right, p=p, r2=r2, dHd=dHd, gd=gd)


def main():
    out = Path(sys.argv[1])
    e1 = json.load(open(out / "e1_profiles.json"))
    kl0 = e1["base_calib"]["kl"]
    print(f"base calib KL {kl0:.5f} nll {e1['base_calib']['nll']:.4f} | test KL {e1['base_test']['kl']:.5f} nll {e1['base_test']['nll']:.4f}\n")
    rows = []
    for label, pr in e1["profiles"].items():
        dkl = [k - kl0 for k in pr["kl"]]
        rows.append((label, classify(e1["alphas"], dkl, pr["g_dot_d"], pr["dHd"])))
    print(f"{'gate':13s} {'class':6s} {'dKL(0)':>8s} {'quad(0)':>8s} {'dKL(.5)':>8s} {'dKL(1.5)':>8s} {'dKL(2)':>8s} {'r_left':>6s} {'r_right':>7s} {'p':>5s} {'R2':>5s} {'dHd':>9s}")
    for label, r in rows:
        print(f"{label:13s} {r['cls']:6s} {1e3*r['d0']:8.2f} {1e3*r['quad0']:8.2f} {1e3*r['d_half']:8.2f} {1e3*r['d_15']:8.2f} {1e3*r['d_2']:8.2f} "
              f"{r['r_left']:6.2f} {r['r_right']:7.2f} {r['p']:5.2f} {r['r2']:5.2f} {r['dHd']:9.2e}")
    print("\nclass counts by kind:")
    kinds = sorted({l.split(".")[0] for l, _ in rows})
    for k in kinds:
        sub = [r["cls"] for l, r in rows if l.split(".")[0] == k]
        print(f"  {k:6s} " + ", ".join(f"{c}:{sub.count(c)}" for c in ("flat", "quad", "cliff", "power", "other") if sub.count(c)))

    if (out / "e1_controls.json").exists():
        c = json.load(open(out / "e1_controls.json"))
        bt = c["base_test"]
        print(f"\ncontrols (test): base nll {bt['nll']:.4f} kl {bt['kl']:.5f}")
        for k, v in c.items():
            if k == "base_test":
                continue
            if "nll" in v:
                print(f"  {k:26s} nll {v['nll']:.4f} ({1e3*(v['nll']-bt['nll']):+7.1f} mnat)  kl {v['kl']:.5f}")
            else:
                print(f"  {k:26s} last-pos nll {np.mean(v['last_pos_nll']):.4f} ({1e3*(np.mean(v['last_pos_nll'])-v['base_last_pos_nll']):+7.1f} mnat vs {v['base_last_pos_nll']:.4f})")

    if (out / "e2_pairs.json").exists():
        e2 = json.load(open(out / "e2_pairs.json"))
        kl0 = e2["base_kl"]; grid = e2["grid"]
        print("\nE2 pairwise non-additivity  R(a,b) = dKL(a,b) - dKL(a,1) - dKL(1,b)   [mnat]")
        recs = []
        for name, pr in e2["pairs"].items():
            M = np.array(pr["kl"]) - kl0
            R = M - M[:, [4]] - M[[4], :]
            # quadratic prediction of the cross term: (a-1)(b-1) d_i H d_j
            A = np.array(grid)[:, None] - 1; B = np.array(grid)[None, :] - 1
            Rq = A * B * pr["cross_hessian"]
            recs.append((name, R[0, 0], np.abs(R).max(), Rq[0, 0], M[0, 4], M[4, 0], M[0, 0]))
        recs.sort(key=lambda r: -abs(r[1]))
        print(f"{'pair':24s} {'R(0,0)':>8s} {'max|R|':>8s} {'quadR(0,0)':>10s} {'dKL_i(0)':>9s} {'dKL_j(0)':>9s} {'dKL(0,0)':>9s}")
        for r in recs[:25]:
            print(f"{r[0]:24s} {1e3*r[1]:8.2f} {1e3*r[2]:8.2f} {1e3*r[3]:10.2f} {1e3*r[4]:9.2f} {1e3*r[5]:9.2f} {1e3*r[6]:9.2f}")
        Rs = np.array([r[1] for r in recs]); Rq = np.array([r[3] for r in recs])
        print(f"\n{len(recs)} pairs: median |R(0,0)| = {1e3*np.median(np.abs(Rs)):.2f} mnat, "
              f"90th pct = {1e3*np.percentile(np.abs(Rs),90):.2f}, max = {1e3*np.abs(Rs).max():.2f}")
        print(f"fraction of pairs with |R(0,0)| > 1 mnat: {(np.abs(Rs)>1e-3).mean():.2f}; "
              f"corr(R, quad cross-term) = {np.corrcoef(Rs, Rq)[0,1]:.2f}")
        heads = [r for r in recs if r[0].startswith("head")]
        if heads:
            Rh = np.array([r[1] for r in heads])
            print(f"head pairs only: median |R| {1e3*np.median(np.abs(Rh)):.2f}, max {1e3*np.abs(Rh).max():.2f}, "
                  f"share of joint dKL(0,0) that is non-additive (median) = {np.median([abs(r[1])/max(abs(r[6]),1e-9) for r in heads]):.2f}")


if __name__ == "__main__":
    main()
