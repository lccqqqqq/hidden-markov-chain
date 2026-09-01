"""FWSVD/SVD sweep: rank ratios x {fisher-weighted, plain} x {no fine-tune, 1-epoch FT}."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch as t
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.fwsvd import FWSVD
from evaluate import evaluate, load_run, load_test_data

RUN = "models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN"
device = "cuda" if t.cuda.is_available() else "cpu"
model, cfg = load_run(RUN, device=device)
test = load_test_data(cfg, "test"); train = load_test_data(cfg, "train")

rows = []
def run(ratio, weighted, epochs):
    q = FWSVD(ratio, weighted=weighted, epochs=epochs)
    qm = q.quantize(model, train)
    r = evaluate(qm, test, device=device)
    n = q.params_after(model)
    rows.append(dict(q.spec.as_row(), params=n, bytes=n * 4, loss=r["loss"],
                     ranks=q.ranks, loss_per_pos=[round(x, 4) for x in r["loss_per_pos"]]))
    print(f"{q.name:6s} rho={ratio:<7} ft={epochs}  params={n:6d}  bytes={n*4:7d}  loss={r['loss']:.4f}", flush=True)

for ratio in (0.75, 0.5, 0.375, 0.25, 0.125, 0.0625):
    for weighted in (True, False):
        run(ratio, weighted, 0)
for ratio in (0.5, 0.25, 0.125, 0.0625):
    for weighted in (True, False):
        run(ratio, weighted, 1)

out = Path("results/compression/L4_d64_fwsvd.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})
print("saved", out)
