"""Composition sweep: FWSVD(+FT) factors quantized by RTN (PTQ) or STE-QAT."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch as t
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.fwsvd import FWSVD
from compress.experimental.lowrank_quant import apply_lowrank_quant, composed_bytes
from evaluate import evaluate, load_run, load_test_data

RUN = "models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN"
device = "cuda" if t.cuda.is_available() else "cpu"
model, cfg = load_run(RUN, device=device)
test = load_test_data(cfg, "test"); train = load_test_data(cfg, "train")

rows = []
for ratio in (0.25, 0.125, 0.0625):
    fw = FWSVD(ratio, weighted=True, epochs=1)
    base = fw.quantize(model, train)          # fine-tuned low-rank model; factors cached
    n_params = fw.params_after(model)
    r0 = evaluate(base, test, device=device)
    print(f"FWSVD+FT rho={ratio}  params={n_params}  loss={r0['loss']:.4f}", flush=True)
    for bits in (8, 4, 3, 2):
        for mode in ("ptq", "qat"):
            if mode == "qat" and bits == 8:
                continue
            qm = apply_lowrank_quant(model, fw.factors, bits, mode, train_data=train)
            r = evaluate(qm, test, device=device)
            b = composed_bytes(model, fw.factors, bits)
            rows.append(dict(method=f"FWSVD+{'RTN' if mode=='ptq' else 'QAT'}", rank_ratio=ratio,
                             bits=bits, params=n_params, bytes=b, loss=r["loss"],
                             loss_per_pos=[round(x, 4) for x in r["loss_per_pos"]]))
            print(f"  rho={ratio} bits={bits} {mode}  bytes={b:6d}  loss={r['loss']:.4f}", flush=True)

out = Path("results/compression/L4_d64_lowrank_quant.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})
print("saved", out)
