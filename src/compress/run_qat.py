"""STE-QAT runs: uniform 4/3/2-bit, plus fine-tuning of HAWQ-V2 allocations (the paper's
prescribed allocation->QAT pipeline). Writes CSV rows compatible with the other sweeps."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch as t
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.base import count_bytes
from compress.hawq import HAWQv2
from compress.qat import STEQAT
from compress.rtn import RTN
from evaluate import evaluate, load_run, load_test_data

RUN = "models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN"
device = "cuda" if t.cuda.is_available() else "cpu"
model, cfg = load_run(RUN, device=device)
test = load_test_data(cfg, "test"); train = load_test_data(cfg, "train")
g = t.Generator().manual_seed(0)
calib128 = train[t.randperm(len(train), generator=g)[:128]]

rows = []
def run(tag, q, bits_for_bytes):
    qm = q.quantize(model, train)
    r = evaluate(qm, test, device=device)
    b = count_bytes(qm, bits_for_bytes)
    rows.append(dict(q.spec.as_row(), tag=tag, bytes=b, loss=r["loss"],
                     loss_per_pos=[round(x, 4) for x in r["loss_per_pos"]]))
    print(f"{tag:28s} bytes={b}  loss={r['loss']:.4f}", flush=True)

for bits in (4, 3, 2):
    run(f"uniform-{bits}bit", STEQAT(bits), bits)

for budget in (95744, 82944):
    h = HAWQv2(budget, RTN(8), choices=(2, 3, 4, 5, 6, 8))
    h.allocate(model, calib128)
    q = STEQAT(8)          # bits value unused when bits_map covers all tensors
    q.bits_map = h.allocation
    run(f"hawq-{budget}B", q, h.allocation)

out = Path("results/compression/L4_d64_qat.csv"); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})
print("saved", out)
