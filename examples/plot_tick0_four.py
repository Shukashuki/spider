"""Plot tick-0 convergence for tick0_four experiment (3 tasks × 4 modes).

Usage:
    python examples/plot_tick0_four.py [--base outputs/tick0_four] [--out outputs/tick0_four/summary.png]

Directory layout expected:
    {base}/{task}_{mode}/trajectory_mjwp.npz
where mode ∈ {dial, mppi, cma, cma_rank, cma_dial}
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


COLOR = {
    "dial":     "steelblue",
    "mppi":     "darkorange",
    "cma":      "forestgreen",   # legacy dir name → displayed as CMA-rank
    "cma_rank": "forestgreen",
    "cma_dial": "purple",
}
LABEL = {
    "dial":     "DIAL",
    "mppi":     "MPPI",
    "cma":      "CMA-rank",
    "cma_rank": "CMA-rank",
    "cma_dial": "CMA-DIAL",
}
MODE_ORDER = ["dial", "mppi", "cma", "cma_rank", "cma_dial"]


def load_tick0(path: str) -> dict | None:
    npz = (os.path.join(path, "trajectory_mjwp_act.npz")
           if os.path.exists(os.path.join(path, "trajectory_mjwp_act.npz"))
           else os.path.join(path, "trajectory_mjwp.npz"))
    if not os.path.exists(npz):
        return None
    d = np.load(npz)
    if "rew_max" not in d or "rew_u0" not in d:
        return None
    rew_max = d["rew_max"][0].astype(float)
    rew_u0  = d["rew_u0"][0].astype(float)
    # mask trailing zeros (early-stopping padding)
    for j in range(len(rew_max) - 1, -1, -1):
        if rew_max[j] == 0.0:
            rew_max[j] = np.nan
        else:
            break
    return {"rew_max": rew_max, "rew_u0": rew_u0}


def parse_dir(name: str):
    """Return (task, mode) from a directory name like 'p36-tea_cma_dial'."""
    for mode in sorted(MODE_ORDER, key=len, reverse=True):  # longest first
        suffix = f"_{mode}"
        if name.endswith(suffix):
            task = name[: -len(suffix)]
            return task, mode
    return name, "dial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/tick0_four")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = args.base
    entries = {}   # task -> list of (mode, data)
    for name in sorted(os.listdir(base)):
        data = load_tick0(os.path.join(base, name))
        if data is None:
            continue
        task, mode = parse_dir(name)
        entries.setdefault(task, []).append((mode, data))

    tasks = sorted(entries.keys())
    if not tasks:
        print(f"No completed runs found in {base}")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5),
                             sharey=False, gridspec_kw={"wspace": 0.35})
    if len(tasks) == 1:
        axes = [axes]

    fig.suptitle(
        "Tick 0 zero-init: DIAL / MPPI / CMA-rank / CMA-DIAL  (contact guidance params)",
        fontsize=11, fontweight="bold",
    )

    print(f"\n{'Tag':<35} {'u0':>8} {'final':>8} {'iters':>7}")
    print("-" * 62)

    for ax, task in zip(axes, tasks):
        mode_list = entries[task]
        # sort by MODE_ORDER for legend consistency
        mode_list.sort(key=lambda x: MODE_ORDER.index(x[0]) if x[0] in MODE_ORDER else 99)

        for mode, data in mode_list:
            rew_max = data["rew_max"]
            u0_val  = float(data["rew_u0"][0])

            valid = ~np.isnan(rew_max)
            x_iter = np.arange(1, valid.sum() + 1)
            y_iter = rew_max[valid]

            x = np.concatenate([[0], x_iter])
            y = np.concatenate([[u0_val], y_iter])

            ax.plot(x, y, label=LABEL.get(mode, mode),
                    color=COLOR.get(mode, "black"),
                    linewidth=1.8, marker="o", markersize=2.5)

            final = float(y_iter[-1]) if len(y_iter) else float("nan")
            iters = int(valid.sum())
            print(f"{task}_{mode:<20} {u0_val:>8.4f} {final:>8.4f} {iters:>7}")

        ax.axhline(float(entries[task][0][1]["rew_u0"][0]),
                   color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Optimizer iter  (0 = u0=0)")
        ax.set_ylabel("rew_max")
        ax.set_title(f"{task}  — tick 0, zero-init")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    out = args.out or os.path.join(base, "summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
