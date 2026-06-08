"""Plot reward convergence across optimizer iterations.

rew_max shape: (T, max_iter) — T control steps, each row is one step's
convergence curve. Zeros = early stopping (optimizer quit before max_iter).

Usage:
    python examples/plot_reward_convergence.py \
        outputs/ablation/p36-tea/baseline/trajectory_mjwp.npz \
        outputs/ablation/p36-tea/cma_mppi/trajectory_mjwp.npz \
        --labels "CEM" "MPPI-CMA" \
        --out outputs/reward_convergence.png
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_rew(path: str) -> np.ndarray:
    """Load rew_max (T, max_iter), mask out zero-padding."""
    d = np.load(path)
    r = d["rew_max"].astype(np.float32)          # (T, max_iter)
    # replace trailing zeros with NaN (early stopping padding)
    for t in range(r.shape[0]):
        row = r[t]
        valid = np.where(row != 0.0)[0]
        if len(valid) > 0:
            last = valid[-1] + 1
            row[last:] = np.nan
    return r  # (T, max_iter), NaN where stopped


def plot(runs: list[tuple[str, np.ndarray]], out: str | None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"wspace": 0.35})

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── Left: mean convergence curve (averaged over T control steps) ──
    ax = axes[0]
    for i, (label, r) in enumerate(runs):
        mean = np.nanmean(r, axis=0)   # (max_iter,)
        std  = np.nanstd(r, axis=0)
        valid = ~np.isnan(mean)
        x = np.arange(r.shape[1])[valid]
        ax.plot(x, mean[valid], label=label, color=colors[i], linewidth=2)
        ax.fill_between(x,
                        (mean - std)[valid],
                        (mean + std)[valid],
                        color=colors[i], alpha=0.15)
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("rew_max")
    ax.set_title("Mean convergence (± std across control steps)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Right: final reward per control step ──
    ax = axes[1]
    for i, (label, r) in enumerate(runs):
        # last non-NaN value per control step = final reward after convergence
        final = np.array([
            row[~np.isnan(row)][-1] if (~np.isnan(row)).any() else np.nan
            for row in r
        ])
        ax.plot(np.arange(len(final)), final,
                label=label, color=colors[i], linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Control step")
    ax.set_ylabel("rew_max (converged)")
    ax.set_title("Final reward per control step  (↑ better)")
    ax.legend()
    ax.grid(alpha=0.3)

    # summary
    print(f"\n{'Optimizer':<20} {'mean final rew':>16} {'mean iters':>12}")
    print("-" * 50)
    for label, r in runs:
        final = np.array([
            row[~np.isnan(row)][-1] if (~np.isnan(row)).any() else np.nan
            for row in r
        ])
        iters = np.array([(~np.isnan(r[t])).sum() for t in range(r.shape[0])])
        print(f"{label:<20} {np.nanmean(final):>16.4f} {iters.mean():>12.1f}")

    fig.suptitle("Reward convergence comparison", fontsize=13, fontweight="bold", y=1.02)
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", nargs="+", help="NPZ files to compare")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    labels = args.labels or [f"run{i}" for i in range(len(args.npz))]
    runs = [(label, load_rew(path)) for label, path in zip(labels, args.npz)]
    plot(runs, args.out)


if __name__ == "__main__":
    main()
