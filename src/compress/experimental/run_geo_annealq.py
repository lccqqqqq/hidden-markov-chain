"""
R1.4 — annealed quantization (relaxation_program.md).

Bit-width annealing 8 -> 6 -> 4 -> 3 -> 2 with N STE-QAT steps per stage (each stage starts
from the previous stage's baked lattice weights) vs direct 2-bit STE-QAT with the same total
number of steps (5N); soft (exact-teacher) targets; per-channel asym grids (ExtSTEQAT with a
uniform bits_map). Two targets: uniform 2-bit (70 272 B) and the E4 measured-sensitivity
allocation at 70 272 B (read from e4_alloc.csv; annealing then means every tensor's width
is lowered stage by stage from 8 towards its final width).

  uv run python -u src/compress/experimental/run_geo_annealq.py models/cylinder_graph/<run> --N 50 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compress.base import is_quantized_param
from compress.experimental.geo_alloc import DELETE, TERNARY, ExtGPTQ, ExtSTEQAT, count_bytes_ext
from compress.experimental.teacher import evaluate_teacher, teacher_probs
from evaluate import load_run, load_test_data
from utils import create_process_from_dict

STAGES = [8, 6, 4, 3, 2]
LABEL2W = {"del": DELETE, "T": TERNARY}


def stage_map(final: dict[str, int], stage_bits: int) -> dict[str, int]:
    """Every tensor at max(final width, stage width); ternary/delete count as 'below 2'."""
    out = {}
    for n, b in final.items():
        if b in (DELETE, TERNARY):
            out[n] = stage_bits if stage_bits > 2 else b
        else:
            out[n] = max(b, stage_bits)
    return out


def anneal_quantize(q: ExtSTEQAT, model, train, proc, final: dict[str, int], N: int):
    """ExtSTEQAT's loop with the bit-width schedule changed every N steps (latents kept)."""
    import copy
    from torch.nn.utils import parametrize
    from compress.experimental.geo_alloc import _ExtSTE
    from compress.experimental.teacher import kl_loss
    model = copy.deepcopy(model)
    wrapped = []
    for name, _ in list(model.named_parameters()):
        if not is_quantized_param(name):
            continue
        mod = model.get_submodule(name.rsplit(".", 1)[0]); attr = name.rsplit(".", 1)[1]
        parametrize.register_parametrization(mod, attr, _ExtSTE(q, name))
        wrapped.append((mod, attr))
    opt = t.optim.Adam(model.parameters(), lr=q.lr)
    g = t.Generator().manual_seed(q.seed)
    model.train()
    for step in range(5 * N):
        q.bits_map = stage_map(final, STAGES[min(step // N, len(STAGES) - 1)])
        idx = t.randint(0, len(train), (q.batch_size,), generator=g)
        batch = train[idx]
        loss = kl_loss(model, batch, teacher_probs(proc, batch))
        opt.zero_grad(); loss.backward(); opt.step()
    q.bits_map = dict(final)
    for mod, attr in wrapped:
        parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
    for name, p in model.named_parameters():
        if is_quantized_param(name) and final.get(name) == DELETE:
            p.data.zero_()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--N", nargs="+", type=int, default=[50, 200])
    ap.add_argument("--out", default="results/compression/geometry")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    t.set_num_threads(args.threads)
    out = Path(args.out)

    model, cfg = load_run(args.run_dir)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    test, train = load_test_data(cfg, "test"), load_test_data(cfg, "train")
    p_test = teacher_probs(proc, test)
    base = evaluate_teacher(model, test, p_test)
    names = [n for n, _ in model.named_parameters() if is_quantized_param(n)]

    targets = {"uniform2": {n: 2 for n in names}}
    e4 = out / "e4_alloc.csv"
    if e4.exists():
        for row in csv.DictReader(open(e4)):
            if row["signal"] == "measured" and row["budget"] == "70272" and row["stage"] == "PTQ":
                alloc = json.loads(row["allocation"])
                targets["measured70272"] = {n: LABEL2W.get(w, int(w) if w not in LABEL2W else w) for n, w in alloc.items()}
                break

    rows = []
    for tname, final in targets.items():
        for N in args.N:
            for mode in ("direct", "anneal"):
                t0 = time.time()
                if mode == "direct":
                    q = ExtSTEQAT(16, max_steps=5 * N, soft_targets=True, process=proc); q.bits_map = final
                    qm = q.quantize(model, train)
                    path = "2-bit x 5N"
                else:
                    # single parametrized run; the width schedule is applied by mutating
                    # bits_map in place (the STE reads it every forward), so the fp32 latent
                    # weights are kept across stages. Baking between stages was tried first
                    # and fails: latents frozen on the coarser lattice cannot flip cells.
                    q = ExtSTEQAT(16, max_steps=5 * N, soft_targets=True, process=proc)
                    q.bits_map = stage_map(final, STAGES[0])
                    qm = anneal_quantize(q, model, train, proc, final, N)
                    path = "->".join(map(str, STAGES)) + " (latents kept)"
                r = evaluate_teacher(qm, test, p_test)
                rows.append(dict(target=tname, N=N, mode=mode, path=path, total_steps=5 * N,
                                 bytes=count_bytes_ext(qm, final), test_nll=r["nll"], test_kl=r["kl"],
                                 d_mnat=1e3 * (r["nll"] - base["nll"]), seconds=round(time.time() - t0)))
                print(f"{tname:14s} N={N:4d} {mode:7s} {path:14s} bytes={rows[-1]['bytes']} "
                      f"test_nll={r['nll']:.4f} (d={rows[-1]['d_mnat']:+.1f} mnat) [{rows[-1]['seconds']}s]", flush=True)
                with open(out / "r1_annealq.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
