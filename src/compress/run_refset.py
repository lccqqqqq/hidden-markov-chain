"""AWQ / LSQ / structured-pruning runs (reference-set completion)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch as t
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.awq import AWQ
from compress.base import count_bytes
from compress.prune import NeuronPrune
from compress.qat import LSQ
from evaluate import evaluate, load_run, load_test_data

RUN = "models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN"
device = "cuda" if t.cuda.is_available() else "cpu"
model, cfg = load_run(RUN, device=device)
test = load_test_data(cfg, "test"); train = load_test_data(cfg, "train")
g = t.Generator().manual_seed(0)
calib = train[t.randperm(len(train), generator=g)[:128]]

rows = []
def add(q, qm, params, bytes_, tag=""):
    r = evaluate(qm, test, device=device)
    rows.append(dict(q.spec.as_row(), tag=tag, params=params, bytes=bytes_, loss=r["loss"],
                     loss_per_pos=[round(x, 4) for x in r["loss_per_pos"]]))
    print(f"{q.name:16s} {tag:12s} params={params}  bytes={bytes_:7d}  loss={r['loss']:.4f}", flush=True)

n_full = sum(p.numel() for p in model.parameters())
for bits in (8, 4, 3, 2):
    q = AWQ(bits)
    add(q, q.quantize(model, calib), n_full, count_bytes(model, bits), f"{bits}bit")
for bits in (4, 3, 2):
    q = LSQ(bits)
    add(q, q.quantize(model, train), n_full, count_bytes(model, bits), f"{bits}bit")
for crit in ("magnitude", "wanda"):
    for frac in (0.25, 0.5, 0.75, 0.875):
        q0 = NeuronPrune(frac, crit, epochs=0)
        qm0 = q0.quantize(model, calib)
        n = q0.params_after(model)
        add(q0, qm0, n, n * 4, f"f={frac} noFT")
        q1 = NeuronPrune(frac, crit, epochs=1)
        add(q1, q1.quantize(model, train), n, n * 4, f"f={frac} FT")

out = Path("results/compression/L4_d64_refset.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})
print("saved", out)
