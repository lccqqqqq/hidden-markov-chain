"""
Plot BPE forward vs reversed training dynamics with seed variance bands.

Reversal is at the BPE token level (same compression ratio), so losses
are directly comparable without normalization.

Usage:
    python scripts/plot_bpe_dynamics.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CURVE_DIR = Path("data/loss_curves/bpe")
FIG_DIR = Path("figures/bpe_experiment")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_curves(prefix):
    """Load all loss curve JSONs matching prefix (e.g. 'fwd' or 'rev')."""
    curves = []
    for f in sorted(CURVE_DIR.glob(f"{prefix}_seed*.json")):
        with open(f) as fh:
            data = json.load(fh)
        steps = np.array([p["step"] for p in data["loss_curve"]])
        losses = np.array([p["val_loss"] for p in data["loss_curve"]])
        curves.append((steps, losses, data.get("model_seed"), data.get("best_val_loss")))
    return curves


def interpolate_to_common_steps(curves):
    """Interpolate all curves to a common step grid."""
    if not curves:
        return None, None, None
    common_steps = curves[0][0]
    all_losses = []
    for steps, losses, _, _ in curves:
        interp_losses = np.interp(common_steps, steps, losses)
        all_losses.append(interp_losses)
    all_losses = np.array(all_losses)
    return common_steps, np.mean(all_losses, axis=0), np.std(all_losses, axis=0)


def main():
    fwd_curves = load_curves("fwd")
    rev_curves = load_curves("rev")

    if not fwd_curves or not rev_curves:
        print(f"Not enough data. Found {len(fwd_curves)} forward, {len(rev_curves)} reversed curves.")
        print(f"Looking in: {CURVE_DIR}")
        return

    print(f"Loaded {len(fwd_curves)} forward, {len(rev_curves)} reversed curves")

    # Print final losses
    print("\nFinal best val losses:")
    for prefix, curves in [("Forward", fwd_curves), ("Reversed", rev_curves)]:
        best_losses = [c[3] for c in curves if c[3] is not None]
        print(f"  {prefix}: {np.mean(best_losses):.4f} +/- {np.std(best_losses):.4f} "
              f"(individual: {[f'{l:.4f}' for l in best_losses]})")

    fwd_steps, fwd_mean, fwd_std = interpolate_to_common_steps(fwd_curves)
    rev_steps, rev_mean, rev_std = interpolate_to_common_steps(rev_curves)

    # --- Single panel: directly comparable loss ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(fwd_steps, fwd_mean, color='C0', label='Forward BPE')
    ax.fill_between(fwd_steps, fwd_mean - fwd_std, fwd_mean + fwd_std, alpha=0.2, color='C0')
    ax.plot(rev_steps, rev_mean, color='C1', label='Reversed BPE')
    ax.fill_between(rev_steps, rev_mean - rev_std, rev_mean + rev_std, alpha=0.2, color='C1')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Val loss (nats/BPE token)')
    ax.set_title('BPE Forward vs Reversed Training Dynamics\n'
                 '(token-level reversal, same compression, 5 seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = FIG_DIR / "bpe_training_dynamics.pdf"
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(FIG_DIR / "bpe_training_dynamics.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
