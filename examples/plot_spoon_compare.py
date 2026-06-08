"""Analyse pick_spoon_bowl 4-optimizer comparison.

Plots:
  1. rew_max per tick (full task, 4 modes)
  2. Object pos/rot error per tick
  3. Tick-0 zero-init convergence (4 modes)
  4. Timing bar chart

Usage:
    python examples/plot_spoon_compare.py \
        --full outputs/spoon_full \
        --tick0 outputs/spoon_tick0 \
        --timing outputs/spoon_timing.txt \
        --ref example_datasets/processed/oakink/xhand/bimanual/pick_spoon_bowl/0/trajectory_kinematic_act.npz \
        --out outputs/spoon_summary.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


MODES      = ["dial", "mppi", "cma_rank", "cma_dial"]
COLOR      = {"dial": "steelblue", "mppi": "darkorange", "cma_rank": "forestgreen", "cma_dial": "purple"}
LABEL      = {"dial": "DIAL", "mppi": "MPPI", "cma_rank": "CMA-rank", "cma_dial": "CMA-DIAL"}
OBJ_POS_TH = 0.1
OBJ_ROT_TH = 0.3
# oakink_act: ctrl_dt=0.05, ref_dt=0.02 → ref_steps_per_ctrl = 0.05/0.02 = 2.5 → round to 2
REF_STEPS_PER_CTRL = 3   # approximate mapping; tune if needed


def find_npz(path):
    for f in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return p
    return None


def load_full(path):
    npz = find_npz(path)
    if npz is None:
        return None
    d = np.load(npz)
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


def load_tick0(path):
    npz = find_npz(path)
    if npz is None:
        return None
    d = np.load(npz)
    if "rew_max" not in d or "rew_u0" not in d:
        return None
    rew_max = d["rew_max"][0].astype(float)
    rew_u0  = d["rew_u0"][0].astype(float)
    for j in range(len(rew_max) - 1, -1, -1):
        if rew_max[j] == 0.0:
            rew_max[j] = np.nan
        else:
            break
    return {"rew_max": rew_max, "rew_u0": rew_u0}


def obj_errors(qpos, ref_qpos, nq_obj=12):
    """Compute right/left object position error at last sim step per tick."""
    n = qpos.shape[0]
    ref_idx = np.clip(np.arange(n) * REF_STEPS_PER_CTRL, 0, len(ref_qpos) - 1)
    r_pos = qpos[:, -1, -12:-9]
    l_pos = qpos[:, -1, -6:-3]
    r_ref = ref_qpos[ref_idx, -12:-9]
    l_ref = ref_qpos[ref_idx, -6:-3]
    r_err = np.linalg.norm(r_pos - r_ref, axis=1)
    l_err = np.linalg.norm(l_pos - l_ref, axis=1)
    # mask fixed objects
    if np.all(np.abs(r_ref) < 1e-4):
        r_err *= 0
    if np.all(np.abs(l_ref) < 1e-4):
        l_err *= 0
    return r_err, l_err


def load_timing(path):
    timing = {}
    if not path or not os.path.exists(path):
        return timing
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tag = parts[0]
                try:
                    timing[tag] = int(parts[1].replace("s", ""))
                except ValueError:
                    pass
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   default="outputs/spoon_full")
    parser.add_argument("--tick0",  default="outputs/spoon_tick0")
    parser.add_argument("--timing", default="outputs/spoon_timing.txt")
    parser.add_argument("--ref",    default="example_datasets/processed/oakink/xhand/bimanual/pick_spoon_bowl/0/trajectory_kinematic_act.npz")
    parser.add_argument("--out",    default="outputs/spoon_summary.png")
    args = parser.parse_args()

    ref_qpos = np.load(args.ref)["qpos"] if os.path.exists(args.ref) else None
    timing   = load_timing(args.timing)

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_rew   = fig.add_subplot(gs[0, 0])
    ax_obj_r = fig.add_subplot(gs[0, 1])
    ax_obj_l = fig.add_subplot(gs[0, 2])
    ax_tick0 = fig.add_subplot(gs[1, 0:2])
    ax_time  = fig.add_subplot(gs[1, 2])

    fig.suptitle("pick_spoon_bowl — 4 optimizer comparison (oakink_act)",
                 fontsize=13, fontweight="bold")

    # ── 1. rew_max per tick ───────────────────────────────────────────────────
    for mode in MODES:
        data = load_full(os.path.join(args.full, mode))
        if data is None:
            continue
        rew = data["rew_max"][:, -1]   # last iter rew per tick
        ax_rew.plot(rew, label=LABEL[mode], color=COLOR[mode], linewidth=1.6)
    ax_rew.set_xlabel("Tick"); ax_rew.set_ylabel("rew_max (last iter)")
    ax_rew.set_title("Reward per tick"); ax_rew.legend(fontsize=9); ax_rew.grid(alpha=0.3)

    # ── 2 & 3. Object pos error per tick ─────────────────────────────────────
    for mode in MODES:
        data = load_full(os.path.join(args.full, mode))
        if data is None or ref_qpos is None:
            continue
        r_err, l_err = obj_errors(data["qpos"], ref_qpos)
        ax_obj_r.plot(r_err, label=LABEL[mode], color=COLOR[mode], linewidth=1.6)
        ax_obj_l.plot(l_err, label=LABEL[mode], color=COLOR[mode], linewidth=1.6)
    for ax, title in [(ax_obj_r, "Right obj pos error"), (ax_obj_l, "Left obj pos error")]:
        ax.axhline(OBJ_POS_TH, color="red", linestyle="--", linewidth=1, label=f"threshold {OBJ_POS_TH}m")
        ax.set_xlabel("Tick"); ax.set_ylabel("pos error (m)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 4. Tick-0 zero-init convergence ──────────────────────────────────────
    for mode in MODES:
        data = load_tick0(os.path.join(args.tick0, mode))
        if data is None:
            continue
        rew_max = data["rew_max"]
        u0_val  = float(data["rew_u0"][0])
        valid   = ~np.isnan(rew_max)
        x = np.concatenate([[0], np.arange(1, valid.sum() + 1)])
        y = np.concatenate([[u0_val], rew_max[valid]])
        ax_tick0.plot(x, y, label=LABEL[mode], color=COLOR[mode], linewidth=1.8,
                      marker="o", markersize=2.5)
    ax_tick0.set_xlabel("Optimizer iter (0 = u0=0)")
    ax_tick0.set_ylabel("rew_max")
    ax_tick0.set_title("Tick-0 zero-init convergence (32 iters)")
    ax_tick0.legend(fontsize=9); ax_tick0.grid(alpha=0.3)

    # ── 5. Timing bar chart ───────────────────────────────────────────────────
    full_tags  = [f"spoon_full_{m}"  for m in MODES]
    tick0_tags = [f"spoon_tick0_{m}" for m in MODES]
    full_times  = [timing.get(t, 0) for t in full_tags]
    tick0_times = [timing.get(t, 0) for t in tick0_tags]

    x_pos = np.arange(len(MODES))
    w = 0.35
    ax_time.bar(x_pos - w/2, full_times,  w, label="Full (200 ticks)", color=[COLOR[m] for m in MODES], alpha=0.8)
    ax_time.bar(x_pos + w/2, tick0_times, w, label="Tick-0 (32 iters)", color=[COLOR[m] for m in MODES], alpha=0.4)
    ax_time.set_xticks(x_pos); ax_time.set_xticklabels([LABEL[m] for m in MODES], fontsize=9)
    ax_time.set_ylabel("Wall time (s)"); ax_time.set_title("Computation time")
    ax_time.legend(fontsize=9); ax_time.grid(alpha=0.3, axis="y")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Mode':<12} {'rew_final':>10} {'r_pos_mean':>12} {'r_pos_max':>11} {'exceed%':>8} {'full_time':>10} {'tick0_time':>11}")
    print("-" * 80)
    for mode in MODES:
        data = load_full(os.path.join(args.full, mode))
        if data is None:
            print(f"{LABEL[mode]:<12}  MISSING")
            continue
        rew_final = float(data["rew_max"][-1, -1]) if "rew_max" in data else float("nan")
        r_err = l_err = np.array([])
        if ref_qpos is not None:
            r_err, l_err = obj_errors(data["qpos"], ref_qpos)
        exceed = np.mean(r_err > OBJ_POS_TH) * 100 if len(r_err) else float("nan")
        ft = timing.get(f"spoon_full_{mode}", "-")
        t0 = timing.get(f"spoon_tick0_{mode}", "-")
        print(f"{LABEL[mode]:<12} {rew_final:>10.4f} {r_err.mean() if len(r_err) else float('nan'):>12.4f}m "
              f"{r_err.max() if len(r_err) else float('nan'):>11.4f}m {exceed:>8.1f}% {str(ft)+('s' if ft!='-' else ''):>10} {str(t0)+('s' if t0!='-' else ''):>11}")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
