"""Quick comparison plot between two optimizer runs.

Usage:
    python examples/compare_optimizers.py \
        --a outputs/baseline/trajectory_mjwp.npz \
        --b outputs/mppi_cma/trajectory_mjwp.npz \
        --labels sampling mppi_cma

    # Or compare any number of runs:
    python examples/compare_optimizers.py \
        --a path/to/run1.npz --b path/to/run2.npz \
        --labels "CEM (baseline)" "MPPI-CMA"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def final_per_step(arr: np.ndarray) -> np.ndarray:
    """Reduce (T, max_iter) → (T,) by taking the last non-zero value per row.

    Handles early stopping: zeros after the optimizer quit are padding,
    so we find the last non-zero column and return that value.
    If the array is already 1-D (or squeezable to 1-D), return as-is.
    """
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr
    # arr is (T, max_iter)
    result = np.empty(arr.shape[0], dtype=arr.dtype)
    for t in range(arr.shape[0]):
        row = arr[t]
        nonzero = np.where(row != 0.0)[0]
        result[t] = row[nonzero[-1]] if len(nonzero) > 0 else row[-1]
    return result


def plot_comparison(runs: list[tuple[str, dict]], out_path: str | None = None):
    labels = [r[0] for r in runs]
    data   = [r[1] for r in runs]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    metrics = [
        ("qpos_dist_mean",  "qpos tracking error (mean over rollouts)", True),
        ("rew_max",         "Best rollout reward per control step",     False),
        ("rew_mean",        "Mean rollout reward per control step",      False),
        ("opt_steps",       "Optimizer iterations per control step",     True),
    ]

    for ax, (key, title, lower_is_better) in zip(axes, metrics):
        for i, (label, d) in enumerate(zip(labels, data)):
            if key not in d:
                ax.text(0.5, 0.5, f"'{key}' not in NPZ", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                break
            y = final_per_step(d[key])
            T = np.arange(len(y))
            ax.plot(T, y, label=label, color=colors[i % len(colors)], linewidth=1.5)
            ax.set_xlabel("Control step")
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        arrow = "↓ better" if lower_is_better else "↑ better"
        ax.set_title(f"{title}  ({arrow})", fontsize=9)

    # Summary table
    fig.text(0.5, 0.97, "Optimizer comparison", ha="center", va="top",
             fontsize=13, fontweight="bold")

    print("\n=== Summary ===")
    print(f"{'Metric':<30}", end="")
    for label in labels:
        print(f"  {label:>18}", end="")
    print()
    print("-" * (30 + 20 * len(labels)))
    for key, title, lower_is_better in metrics:
        print(f"{key:<30}", end="")
        for d in data:
            if key in d:
                val = final_per_step(d[key])
                mean_val = float(val.mean())
                print(f"  {mean_val:>18.4f}", end="")
            else:
                print(f"  {'N/A':>18}", end="")
        print()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a",      required=True, help="NPZ file for run A")
    parser.add_argument("--b",      required=True, help="NPZ file for run B")
    parser.add_argument("--labels", nargs=2, default=["A", "B"])
    parser.add_argument("--out",    default=None,  help="Save figure to path instead of showing")
    args = parser.parse_args()

    runs = [
        (args.labels[0], load(args.a)),
        (args.labels[1], load(args.b)),
    ]
    plot_comparison(runs, out_path=args.out)


if __name__ == "__main__":
    main()
