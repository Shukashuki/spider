#!/usr/bin/env python3
"""Plot 4-way closed-loop comparison: DIAL-MPC / Pure MPPI / MPPI+CMA anneal / MPPI+CMA no-anneal."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "outputs" / "full_task_compare"

OPTIMIZERS = {
    "dial_mpc": {
        "label": "DIAL-MPC (annealing)",
        "color": "#4C72B0",
        "linestyle": "-",
    },
    "mppi_pure": {
        "label": "Pure MPPI (no annealing)",
        "color": "#DD8452",
        "linestyle": "-",
    },
    "mppi_cma_rank": {
        "label": "MPPI+CMA (rank, anneal σ₀=0.15)",
        "color": "#C44E52",
        "linestyle": "-",
    },
    "mppi_cma_rank_noanneal_s005": {
        "label": "MPPI+CMA (no anneal, σ₀=0.05)",
        "color": "#DAA520",
        "linestyle": "--",
    },
}

N_SEEDS = 3


def load_all():
    all_data = {}
    for opt_name in OPTIMIZERS:
        seeds = []
        for s in range(N_SEEDS):
            p = BASE / opt_name / f"seed_{s}" / "trajectory_mjwp.npz"
            if p.exists():
                seeds.append(dict(np.load(p, allow_pickle=True)))
        if seeds:
            all_data[opt_name] = seeds
    return all_data


def get_final_per_tick(seeds_data, metric):
    """For each seed, get the final-iteration value of `metric` at each tick."""
    n_ticks = min(d[metric].shape[0] for d in seeds_data)
    vals = np.array([
        [d[metric][t, int(d["opt_steps"][t].item()) - 1] for t in range(n_ticks)]
        for d in seeds_data
    ])
    return vals  # (n_seeds, n_ticks)


def plot_fig1_per_tick(all_data, out_dir):
    """3-panel: rew_u0, rew_max, rew_mean per tick."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    metrics = [
        ("rew_u0", "Exploit Reward (rew_u0)"),
        ("rew_max", "Max Reward (rew_max)"),
        ("rew_mean", "Mean Reward (rew_mean)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for opt_name, meta in OPTIMIZERS.items():
            if opt_name not in all_data:
                continue
            vals = get_final_per_tick(all_data[opt_name], metric)
            m, s = vals.mean(0), vals.std(0)
            ticks = np.arange(len(m))
            ax.plot(ticks, m, color=meta["color"], label=meta["label"],
                    linewidth=2, linestyle=meta["linestyle"])
            ax.fill_between(ticks, m - s, m + s, color=meta["color"], alpha=0.12)
        ax.set_xlabel("Tick", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Closed-Loop Per-Tick Performance (3 seeds, 32 iters, 1024 samples)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_per_tick_4way.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig1_per_tick_4way.png")


def plot_fig2_convergence(all_data, out_dir):
    """Convergence curves at early/mid/late ticks."""
    sample = list(all_data.values())[0][0]
    n_ticks = sample["rew_u0"].shape[0]
    n_iters = sample["rew_u0"].shape[1]
    focus = [0, n_ticks // 2, n_ticks - 1]
    labels = ["Early (tick 0)", f"Mid (tick {n_ticks//2})", f"Late (tick {n_ticks-1})"]

    fig, axes = plt.subplots(1, len(focus), figsize=(6.5 * len(focus), 5.5))
    for ax, tick, lbl in zip(axes, focus, labels):
        for opt_name, meta in OPTIMIZERS.items():
            if opt_name not in all_data:
                continue
            curves = []
            for d in all_data[opt_name]:
                row = d["rew_u0"][tick, :].copy().astype(float)
                n_valid = int(d["opt_steps"][tick].item())
                row[n_valid:] = np.nan
                curves.append(row)
            curves = np.array(curves)
            m = np.nanmean(curves, 0)
            s = np.nanstd(curves, 0)
            valid = ~np.isnan(m)
            iters = np.arange(n_iters)
            ax.plot(iters[valid], m[valid], color=meta["color"], label=meta["label"],
                    linewidth=2, linestyle=meta["linestyle"])
            ax.fill_between(iters[valid], (m - s)[valid], (m + s)[valid],
                            color=meta["color"], alpha=0.12)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("rew_u0 (exploit)", fontsize=11)
        ax.set_title(lbl, fontsize=12)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Within-Tick Convergence — rew_u0 (Closed-Loop)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_convergence_4way.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig2_convergence_4way.png")


def plot_fig3_tracking(all_data, out_dir):
    """Object tracking error per tick."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for opt_name, meta in OPTIMIZERS.items():
        if opt_name not in all_data:
            continue
        vals = get_final_per_tick(all_data[opt_name], "qpos_dist_mean")
        m, s = vals.mean(0), vals.std(0)
        ticks = np.arange(len(m))
        ax.plot(ticks, m, color=meta["color"], label=meta["label"],
                linewidth=2, linestyle=meta["linestyle"])
        ax.fill_between(ticks, m - s, m + s, color=meta["color"], alpha=0.12)
    ax.set_xlabel("Tick", fontsize=11)
    ax.set_ylabel("qpos_dist_mean", fontsize=11)
    ax.set_title("Object Tracking Error (qpos distance) per Tick", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_tracking_4way.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig3_tracking_4way.png")


def plot_fig4_improvement(all_data, out_dir):
    """Reward improvement (iter0 → iter31) per tick — shows how much each optimizer gains from optimization."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for opt_name, meta in OPTIMIZERS.items():
        if opt_name not in all_data:
            continue
        seeds_data = all_data[opt_name]
        n_ticks = min(d["rew_u0"].shape[0] for d in seeds_data)
        # improvement = final_iter - iter0 (higher = more improvement)
        improvement = np.array([
            [d["rew_u0"][t, int(d["opt_steps"][t].item()) - 1] - d["rew_u0"][t, 0]
             for t in range(n_ticks)]
            for d in seeds_data
        ])
        m, s = improvement.mean(0), improvement.std(0)
        ticks = np.arange(n_ticks)
        ax.plot(ticks, m, color=meta["color"], label=meta["label"],
                linewidth=2, linestyle=meta["linestyle"])
        ax.fill_between(ticks, m - s, m + s, color=meta["color"], alpha=0.12)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Tick", fontsize=11)
    ax.set_ylabel("Δ rew_u0 (iter31 − iter0)", fontsize=11)
    ax.set_title("Per-Tick Optimization Gain (how much reward improves over 32 iters)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_improvement_4way.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig4_improvement_4way.png")


def print_summary(all_data):
    print("\n" + "=" * 95)
    print(f"{'Optimizer':<38} {'Avg Final Rew':>16} {'Avg qpos_dist':>16} {'Avg Δrew':>12}")
    print("-" * 95)
    for opt_name, meta in OPTIMIZERS.items():
        if opt_name not in all_data:
            continue
        seeds_data = all_data[opt_name]
        n_ticks = min(d["rew_u0"].shape[0] for d in seeds_data)

        final_rews = np.array([
            np.mean([d["rew_u0"][t, int(d["opt_steps"][t].item()) - 1] for t in range(n_ticks)])
            for d in seeds_data
        ])
        final_dists = np.array([
            np.mean([d["qpos_dist_mean"][t, int(d["opt_steps"][t].item()) - 1] for t in range(n_ticks)])
            for d in seeds_data
        ])
        delta_rews = np.array([
            np.mean([d["rew_u0"][t, int(d["opt_steps"][t].item()) - 1] - d["rew_u0"][t, 0]
                     for t in range(n_ticks)])
            for d in seeds_data
        ])
        print(f"{meta['label']:<38} {final_rews.mean():>7.4f} ± {final_rews.std():.4f}"
              f"   {final_dists.mean():>7.4f} ± {final_dists.std():.4f}"
              f"   {delta_rews.mean():>+7.4f}")
    print("=" * 95)


if __name__ == "__main__":
    all_data = load_all()
    out = BASE
    print(f"Loaded: {list(all_data.keys())}")
    plot_fig1_per_tick(all_data, out)
    plot_fig2_convergence(all_data, out)
    plot_fig3_tracking(all_data, out)
    plot_fig4_improvement(all_data, out)
    print_summary(all_data)
