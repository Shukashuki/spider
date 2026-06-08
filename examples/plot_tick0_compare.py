"""Plot tick-0 convergence for all experiments under outputs/tick0_compare/.

Usage:
    python examples/plot_tick0_compare.py [--base outputs/tick0_compare] [--out outputs/tick0_compare/summary.png]
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def load_tick0(path: str) -> dict | None:
    """Load tick-0 row from trajectory_mjwp.npz. Returns None if missing."""
    npz = os.path.join(path, "trajectory_mjwp.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz)
    row = {}
    for key in ("rew_max", "rew_u0"):
        if key not in d:
            return None
        arr = d[key][0].astype(float)  # tick 0, shape (max_iter,)
        # mask trailing zeros (early stopping padding)
        for j in range(len(arr) - 1, -1, -1):
            if arr[j] == 0.0:
                arr[j] = np.nan
            else:
                break
        row[key] = arr
    return row


def parse_tag(tag: str):
    """Extract (mode, horizon, sigma) from directory name."""
    mode = "dial" if tag.startswith("dial") else "mppi" if tag.startswith("mppi") else "cma_rank" if tag.startswith("cma_rank") else "cma_dial"
    h_match = re.search(r"h(\d+_\d+|\d+)", tag)
    if h_match:
        h_str = h_match.group(1)
        if "_" in h_str:
            horizon = float(h_str.replace("_", "."))       # "0_8" → 0.8
        else:
            horizon = float(h_str[0] + "." + h_str[1:])   # "08" → 0.8
    else:
        horizon = -1.0
    s_match = re.search(r"s(\d+)", tag)
    sigma = float("0." + s_match.group(1)) if s_match else None
    return mode, horizon, sigma


def label_for(mode, horizon, sigma):
    s = f"σ₀={sigma}" if sigma is not None else ""
    return f"{mode} H={horizon} {s}".strip()


COLOR = {"dial": "steelblue", "mppi": "darkorange", "cma_rank": "forestgreen", "cma_dial": "purple"}
LINESTYLE = {0.8: "--", 1.6: "-"}


def load_ik_u0(ref_base: str) -> dict[float, float]:
    """Load IK-reference u0 rewards per horizon from a reference-init run dir."""
    ik_u0 = {}
    if not os.path.isdir(ref_base):
        return ik_u0
    for tag in os.listdir(ref_base):
        npz = os.path.join(ref_base, tag, "trajectory_mjwp.npz")
        if not os.path.exists(npz):
            continue
        _, horizon, sigma = parse_tag(tag)
        if horizon <= 0 or sigma is not None:  # skip cma variants (same u0)
            continue
        d = np.load(npz)
        if "rew_u0" not in d:
            continue
        u0_val = float(d["rew_u0"][0, 0])
        if horizon not in ik_u0:
            ik_u0[horizon] = u0_val
    return ik_u0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/tick0_compare")
    parser.add_argument("--ref-base", default="outputs/tick0_compare",
                        help="Reference-init run dir to extract IK u0 rewards")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = args.base
    ref_base = args.ref_base
    tags = sorted(os.listdir(base)) if os.path.isdir(base) else []
    entries = []
    for tag in tags:
        data = load_tick0(os.path.join(base, tag))
        if data is None:
            continue
        mode, horizon, sigma = parse_tag(tag)
        entries.append((tag, mode, horizon, sigma, data))

    if not entries:
        print(f"No completed runs found in {base}")
        return

    # IK reference u0 rewards (from reference-init runs)
    ik_u0 = load_ik_u0(ref_base)
    if ik_u0:
        print("IK reference u0 rewards:", {f"h={h}": f"{v:.4f}" for h, v in ik_u0.items()})

    # ── figure layout: one column per horizon ─────────────────────────────────
    horizons = sorted(set(e[2] for e in entries))
    fig, axes = plt.subplots(1, len(horizons), figsize=(7 * len(horizons), 5),
                             sharey=False, gridspec_kw={"wspace": 0.3})
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        h_entries = [e for e in entries if e[2] == h]

        # IK reference u0 horizontal line
        if h in ik_u0:
            ax.axhline(ik_u0[h], color="black", linestyle=":", linewidth=1.5,
                       label=f"IK ref u0  ({ik_u0[h]:.4f})", zorder=5)

        # group by mode for colour consistency
        for tag, mode, horizon, sigma, data in h_entries:
            rew_max = data["rew_max"]
            u0_val  = float(data["rew_u0"][0])  # u0 at iter=0

            valid = ~np.isnan(rew_max)
            x_iter = np.arange(1, valid.sum() + 1)
            y_iter = rew_max[valid]

            x = np.concatenate([[0], x_iter])
            y = np.concatenate([[u0_val], y_iter])

            lbl = label_for(mode, horizon, sigma)
            ls = "-" if sigma is None else (
                "-" if sigma <= 0.05 else ("--" if sigma <= 0.1 else ":")
            )
            ax.plot(x, y, label=lbl, color=COLOR[mode], linestyle=ls, linewidth=1.8,
                    marker="o", markersize=2.5)

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Optimizer iteration  (0 = u0)")
        ax.set_ylabel("rew_max")
        ax.set_title(f"Tick 0 convergence  |  horizon={h}s")
        ax.legend(fontsize=8, ncol=1)
        ax.grid(alpha=0.3)

    fig.suptitle("gigahand p36-tea  — tick 0 optimizer comparison (zero init vs IK ref)",
                 fontsize=12, fontweight="bold", y=1.02)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Tag':<30} {'u0':>8} {'final':>8} {'iters':>7}")
    print("-" * 58)
    for tag, mode, horizon, sigma, data in entries:
        u0   = float(data["rew_u0"][0])
        valid = data["rew_max"][~np.isnan(data["rew_max"])]
        final = float(valid[-1]) if len(valid) else float("nan")
        iters = int((~np.isnan(data["rew_max"])).sum())
        ik = f"  (IK={ik_u0[horizon]:.4f})" if horizon in ik_u0 else ""
        print(f"{tag:<30} {u0:>8.4f} {final:>8.4f} {iters:>7}{ik}")

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
