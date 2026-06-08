"""Visualize MPPI vs MPPI-CMA optimizer convergence (3 seeds).

Generates:
1. Per-iteration reward convergence (mean ± std across seeds, per tick)
2. Improvement heatmaps (seed-averaged)
3. Object z trajectory comparison
4. Convergence speed summary (iterations to 90% of final reward)

Usage:
    cd spider/
    .venv/bin/python experiments/viz_ablation_optimizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "ablation_optimizer", "p36-tea")
SEEDS = [0, 1, 2]

CONDITIONS = {
    "mppi": {"label": "Spider-Origin-Optimizer (MPPI)", "color": "#e74c3c"},
    "mppi_cma": {"label": "MPPI-CMA", "color": "#3498db"},
}

# Load config
with open(os.path.join(OUT, "mppi_seed0", "config_act.yaml")) as f:
    cfg = yaml.safe_load(f)
nq = cfg["nq"]
nq_obj = cfg["nq_obj"]
obj_z_idx = nq - nq_obj + 2

# Load reference
ref = np.load(os.path.join(
    BASE, "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/trajectory_kinematic_act.npz"
))
ref_qpos = ref["qpos"]
ctrl_steps = 40


def load_seeds(cond):
    all_data = []
    for seed in SEEDS:
        path = os.path.join(OUT, f"{cond}_seed{seed}", "trajectory_mjwp_act.npz")
        all_data.append(dict(np.load(path)))
    return all_data


all_data = {}
for cond in CONDITIONS:
    all_data[cond] = load_seeds(cond)

n_ticks = all_data["mppi"][0]["qpos"].shape[0]
n_iters = all_data["mppi"][0]["rew_mean"].shape[1]

ref_z_ticks = []
for t in range(n_ticks):
    ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
    ref_z_ticks.append(ref_qpos[ref_step, obj_z_idx])
ref_z_ticks = np.array(ref_z_ticks)

# ── Figure 1: Per-iteration reward convergence per tick ──
fig, axes = plt.subplots(1, n_ticks, figsize=(5 * n_ticks, 5), sharey=True)
if n_ticks == 1:
    axes = [axes]

for tick in range(n_ticks):
    ax = axes[tick]
    for cond, meta in CONDITIONS.items():
        seeds = all_data[cond]
        rew = np.stack([d["rew_mean"][tick] for d in seeds])  # (3, n_iters)
        m, s = rew.mean(0), rew.std(0)
        ax.plot(range(n_iters), m, color=meta["color"], linewidth=2, label=meta["label"])
        ax.fill_between(range(n_iters), m - s, m + s, color=meta["color"], alpha=0.15)
    ax.set_title(f"Tick {tick}", fontsize=12)
    ax.set_xlabel("Iteration")
    if tick == 0:
        ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    if tick == 0:
        ax.legend(fontsize=8)

plt.suptitle("Reward Convergence: Spider-Origin-Optimizer vs MPPI-CMA (mean ± std, 3 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_reward_convergence.png"), dpi=150, bbox_inches="tight")
print("Saved fig1_reward_convergence.png")

# ── Figure 2: Reward components breakdown per tick ──
components = [
    ("qpos_rew_mean", "qpos", "#e74c3c"),
    ("qvel_rew_mean", "qvel", "#3498db"),
    ("contact_rew_mean", "contact", "#2ecc71"),
]

fig, axes = plt.subplots(n_ticks, 2, figsize=(14, 4 * n_ticks))
if n_ticks == 1:
    axes = axes[np.newaxis, :]

for tick in range(n_ticks):
    for col, (cond, meta) in enumerate(CONDITIONS.items()):
        ax = axes[tick, col]
        seeds = all_data[cond]

        # Total
        rew = np.stack([d["rew_mean"][tick] for d in seeds])
        m, s = rew.mean(0), rew.std(0)
        ax.plot(range(n_iters), m, color="black", linewidth=2, label="total")
        ax.fill_between(range(n_iters), m - s, m + s, color="black", alpha=0.08)

        # Components
        for key, comp_label, comp_color in components:
            vals = np.stack([d[key][tick] for d in seeds])
            m_c = vals.mean(0)
            ax.plot(range(n_iters), m_c, color=comp_color, linewidth=1, alpha=0.8, label=comp_label)

        if tick == 0:
            ax.set_title(meta["label"], fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Tick {tick}", fontsize=10)
        if tick == n_ticks - 1:
            ax.set_xlabel("Iteration")
        if tick == 0 and col == 0:
            ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)

plt.suptitle("Reward Component Breakdown per Tick", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_reward_components.png"), dpi=150, bbox_inches="tight")
print("Saved fig2_reward_components.png")

# ── Figure 3: Improvement heatmaps ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, (cond, meta) in enumerate(CONDITIONS.items()):
    ax = axes[col]
    imp = np.stack([d["improvement"] for d in all_data[cond]]).mean(0)  # (n_ticks, n_iters)
    vmax = max(abs(imp.min()), abs(imp.max()), 0.01)
    im = ax.imshow(imp, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tick")
    ax.set_title(meta["label"])
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Improvement Heatmap (mean over 3 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_improvement_heatmap.png"), dpi=150, bbox_inches="tight")
print("Saved fig3_improvement_heatmap.png")

# ── Figure 4: Object z trajectory + convergence speed ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Object z
ax = axes[0]
ax.plot(range(n_ticks), ref_z_ticks, "k--", linewidth=2, label="Reference", zorder=10)
for cond, meta in CONDITIONS.items():
    seeds_z = np.stack([d["qpos"][:, -1, obj_z_idx] for d in all_data[cond]])
    m, s = seeds_z.mean(0), seeds_z.std(0)
    ax.plot(range(n_ticks), m, "-o", color=meta["color"], label=meta["label"], markersize=5)
    ax.fill_between(range(n_ticks), m - s, m + s, color=meta["color"], alpha=0.15)
ax.set_xlabel("Tick")
ax.set_ylabel("Object z (m)")
ax.set_title("Object Height Trajectory")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Convergence speed: iterations to reach 90% of final reward
ax = axes[1]
for cond, meta in CONDITIONS.items():
    conv_iters = []
    for seed_data in all_data[cond]:
        for tick in range(n_ticks):
            rew_curve = seed_data["rew_mean"][tick]
            final_rew = rew_curve[-1]
            init_rew = rew_curve[0]
            if abs(final_rew - init_rew) < 1e-8:
                conv_iters.append(0)
            else:
                threshold = init_rew + 0.9 * (final_rew - init_rew)
                # Find first iteration crossing threshold
                if final_rew < init_rew:  # reward decreasing (more negative)
                    crossed = np.where(rew_curve <= threshold)[0]
                else:
                    crossed = np.where(rew_curve >= threshold)[0]
                conv_iters.append(crossed[0] if len(crossed) > 0 else n_iters)
    ax.hist(conv_iters, bins=range(n_iters + 1), alpha=0.5, color=meta["color"],
            label=f"{meta['label']} (median={np.median(conv_iters):.0f})", edgecolor="white")
ax.set_xlabel("Iterations to 90% of Final Reward")
ax.set_ylabel("Count (ticks × seeds)")
ax.set_title("Convergence Speed Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_obj_and_convergence.png"), dpi=150)
print("Saved fig4_obj_and_convergence.png")

# ── Figure 5: Min-max reward spread (sample diversity) ──
fig, axes = plt.subplots(1, n_ticks, figsize=(5 * n_ticks, 4), sharey=True)
if n_ticks == 1:
    axes = [axes]

for tick in range(n_ticks):
    ax = axes[tick]
    for cond, meta in CONDITIONS.items():
        seeds = all_data[cond]
        spread = np.stack([d["rew_max"][tick] - d["rew_min"][tick] for d in seeds])
        m, s = spread.mean(0), spread.std(0)
        ax.plot(range(n_iters), m, color=meta["color"], linewidth=2, label=meta["label"])
        ax.fill_between(range(n_iters), m - s, m + s, color=meta["color"], alpha=0.15)
    ax.set_title(f"Tick {tick}")
    ax.set_xlabel("Iteration")
    if tick == 0:
        ax.set_ylabel("Reward Spread (max - min)")
    ax.grid(True, alpha=0.3)
    if tick == 0:
        ax.legend(fontsize=8)

plt.suptitle("Sample Diversity: Reward Spread per Iteration", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_reward_spread.png"), dpi=150, bbox_inches="tight")
print("Saved fig5_reward_spread.png")

print(f"\nDone! All figures saved to: {OUT}")
