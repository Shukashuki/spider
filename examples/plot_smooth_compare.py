"""Compare cma_rank smoothing variants: ctrl jitter, obj error, rew per tick.

Usage:
    python examples/plot_smooth_compare.py --base outputs/spoon_smooth \
        --ref example_datasets/processed/oakink/xhand/bimanual/pick_spoon_bowl/0/trajectory_kinematic_act.npz
"""
from __future__ import annotations
import argparse, os
import matplotlib.pyplot as plt
import numpy as np

COLOR = {"w0": "steelblue", "w5": "darkorange", "w11": "forestgreen"}
LABEL = {"w0": "no smooth", "w5": "smooth=5", "w11": "smooth=11"}
REF_STEPS_PER_CTRL = 3  # oakink_act: ctrl_dt/ref_dt ≈ 0.05/0.02


def find_npz(path):
    for n in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        p = os.path.join(path, n)
        if os.path.exists(p): return p
    return None


def load(run_dir):
    p = find_npz(run_dir)
    if p is None: return None
    return dict(np.load(p))


def ctrl_jitter(ctrl):
    """Mean absolute tick-boundary jump (ctrl[-1] of tick t vs ctrl[0] of tick t+1)."""
    ends   = ctrl[:-1, -1, :]  # (T-1, nu)
    starts = ctrl[1:,   0, :]  # (T-1, nu)
    return np.abs(starts - ends).mean()


def within_jitter(ctrl):
    """Mean absolute step-to-step change within a tick."""
    return np.abs(np.diff(ctrl, axis=1)).mean()


def obj_pos_err(qpos, ref_qpos):
    n = qpos.shape[0]
    ref_idx = np.clip(np.arange(n) * REF_STEPS_PER_CTRL, 0, len(ref_qpos) - 1)
    r_err = np.linalg.norm(qpos[:, -1, -12:-9] - ref_qpos[ref_idx, -12:-9], axis=1)
    if np.all(np.abs(ref_qpos[ref_idx, -12:-9]) < 1e-4): r_err *= 0
    return r_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",  default="outputs/spoon_smooth")
    ap.add_argument("--ref",   default="example_datasets/processed/oakink/xhand/bimanual/pick_spoon_bowl/0/trajectory_kinematic_act.npz")
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()

    ref_qpos = np.load(args.ref)["qpos"] if os.path.exists(args.ref) else None
    variants = [d for d in sorted(os.listdir(args.base))
                if os.path.isdir(os.path.join(args.base, d)) and find_npz(os.path.join(args.base, d))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"wspace": 0.35})
    ax_rew, ax_obj, ax_jitter = axes
    fig.suptitle("CMA-rank ctrl smoothing comparison (pick_spoon_bowl)", fontsize=12, fontweight="bold")

    print(f"\n{'Variant':<12} {'rew_final':>10} {'r_pos_mean':>12} {'tick_jump':>11} {'within_jump':>13}")
    print("-" * 62)

    for tag in variants:
        data = load(os.path.join(args.base, tag))
        if data is None: continue
        c = COLOR.get(tag, "gray"); lbl = LABEL.get(tag, tag)

        # rew per tick
        rew = data["rew_max"][:, -1]
        ax_rew.plot(rew, label=lbl, color=c, linewidth=1.6)

        # obj error per tick
        if ref_qpos is not None:
            r_err = obj_pos_err(data["qpos"], ref_qpos)
            ax_obj.plot(r_err, label=lbl, color=c, linewidth=1.6)

        # jitter stats
        ctrl = data["ctrl"]
        tj = ctrl_jitter(ctrl)
        wj = within_jitter(ctrl)
        rew_final = float(rew[-1])
        r_mean = float(obj_pos_err(data["qpos"], ref_qpos).mean()) if ref_qpos is not None else float("nan")
        print(f"{tag:<12} {rew_final:>10.4f} {r_mean:>12.4f}m {tj:>11.5f} {wj:>13.5f}")

        ax_jitter.bar(tag, tj,  color=c, alpha=0.85, label=f"{lbl} tick-jump")
        ax_jitter.bar(tag, wj,  color=c, alpha=0.35, bottom=tj)

    for ax, title, ylabel in [
        (ax_rew,    "Reward per tick (last iter)", "rew_max"),
        (ax_obj,    "Right obj pos error per tick", "pos error (m)"),
        (ax_jitter, "Ctrl jitter (dark=tick-jump, light=within)", "mean |Δctrl| (rad)"),
    ]:
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        if ax is not ax_jitter: ax.set_xlabel("Tick"); ax.legend(fontsize=9)

    ax_obj.axhline(0.1, color="red", linestyle="--", linewidth=1, label="threshold 0.1m")
    ax_obj.legend(fontsize=8)

    out = args.out or os.path.join(args.base, "smooth_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
