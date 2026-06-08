"""Visualize ablation results across 3 seeds (mean ± std).

Generates:
1. Object z trajectory (mean ± std across seeds)
2. Object position tracking error (mean ± std)
3. Per-iteration reward curves at critical ticks
4. Improvement heatmaps (averaged across seeds)

Usage:
    cd spider/
    .venv/bin/python experiments/viz_ablation_seeds.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "ablation_contact", "p36-tea")
SEEDS = [0, 1, 2]

CONDITIONS = {
    "A_original": {"suffix": "_act", "label": "A: gain decay", "color": "#e74c3c"},
    "B_no_gain_decay": {"suffix": "", "label": "B: nothing", "color": "#3498db"},
    "C_gain_decay_plus_cost": {"suffix": "_act", "label": "C: decay+cost", "color": "#2ecc71"},
    "D_cost_only": {"suffix": "", "label": "D: cost only", "color": "#f39c12"},
}

# Load config
with open(os.path.join(OUT, "A_original_seed0", "config_act.yaml")) as f:
    cfg = yaml.safe_load(f)
nq = cfg["nq"]
nq_obj = cfg["nq_obj"]
obj_z_idx = nq - nq_obj + 2  # z component of object position

# Load reference
ref = np.load(os.path.join(
    BASE, "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/trajectory_kinematic_act.npz"
))
ref_qpos = ref["qpos"]
ctrl_steps = 40


def load_seeds(cond, suffix):
    """Load npz data for all seeds of a condition."""
    all_data = []
    for seed in SEEDS:
        path = os.path.join(OUT, f"{cond}_seed{seed}", f"trajectory_mjwp{suffix}.npz")
        all_data.append(dict(np.load(path)))
    return all_data


def get_obj_z(qpos_array):
    """Extract final-substep object z for each tick."""
    return qpos_array[:, -1, obj_z_idx]


def get_obj_pos_error(qpos_array, n_ticks):
    """Compute object position error per tick."""
    errs = []
    for t in range(n_ticks):
        ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
        sim_obj = qpos_array[t, -1, nq - nq_obj:nq - nq_obj + 3]
        ref_obj = ref_qpos[ref_step, nq - nq_obj:nq - nq_obj + 3]
        errs.append(np.linalg.norm(sim_obj - ref_obj))
    return np.array(errs)


# Load all data
all_data = {}
for cond, meta in CONDITIONS.items():
    all_data[cond] = load_seeds(cond, meta["suffix"])

n_ticks = all_data["A_original"][0]["qpos"].shape[0]
n_iters = all_data["A_original"][0]["rew_mean"].shape[1]

# Reference z at tick boundaries
ref_z_ticks = []
for t in range(n_ticks):
    ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
    ref_z_ticks.append(ref_qpos[ref_step, obj_z_idx])
ref_z_ticks = np.array(ref_z_ticks)

# ── Figure 1: Object z trajectory + tracking error (mean ± std) ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(range(n_ticks), ref_z_ticks, "k--", linewidth=2, label="Reference", zorder=10)
for cond, meta in CONDITIONS.items():
    seeds_z = np.stack([get_obj_z(d["qpos"]) for d in all_data[cond]])  # (3, n_ticks)
    mean_z = seeds_z.mean(axis=0)
    std_z = seeds_z.std(axis=0)
    ax.plot(range(n_ticks), mean_z, "-o", color=meta["color"], label=meta["label"], markersize=4)
    ax.fill_between(range(n_ticks), mean_z - std_z, mean_z + std_z, color=meta["color"], alpha=0.15)
ax.set_ylabel("Object z (m)")
ax.set_title("Object Height Trajectory (mean ± std, 3 seeds)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for cond, meta in CONDITIONS.items():
    seeds_err = np.stack([get_obj_pos_error(d["qpos"], n_ticks) for d in all_data[cond]])
    mean_err = seeds_err.mean(axis=0)
    std_err = seeds_err.std(axis=0)
    ax.plot(range(n_ticks), mean_err, "-o", color=meta["color"], label=meta["label"], markersize=4)
    ax.fill_between(range(n_ticks), mean_err - std_err, mean_err + std_err, color=meta["color"], alpha=0.15)
ax.set_ylabel("Object Position Error (m)")
ax.set_xlabel("Tick")
ax.set_title("Object Tracking Error per Tick (mean ± std, 3 seeds)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_seeds_obj_trajectory.png"), dpi=150)
print("Saved fig1_seeds_obj_trajectory.png")

# ── Figure 2: Per-iteration reward curves at critical ticks (seed-averaged) ──
# Pick critical ticks: early (2), mid-lift (6, 10), late (14)
critical_ticks = [2, 6, 10, 14]

fig, axes = plt.subplots(len(critical_ticks), 4, figsize=(20, 3.5 * len(critical_ticks)))

for row, tick in enumerate(critical_ticks):
    for col, (cond, meta) in enumerate(CONDITIONS.items()):
        ax = axes[row, col]
        seeds = all_data[cond]

        # Stack across seeds: (3, n_iters)
        rew_means = np.stack([d["rew_mean"][tick] for d in seeds])
        qpos_means = np.stack([d["qpos_rew_mean"][tick] for d in seeds])
        qvel_means = np.stack([d["qvel_rew_mean"][tick] for d in seeds])
        contact_means = np.stack([d["contact_rew_mean"][tick] for d in seeds])

        iters = range(n_iters)
        # Total reward
        m, s = rew_means.mean(0), rew_means.std(0)
        ax.plot(iters, m, color="black", linewidth=1.5, label="total")
        ax.fill_between(iters, m - s, m + s, color="black", alpha=0.1)
        # qpos
        m, s = qpos_means.mean(0), qpos_means.std(0)
        ax.plot(iters, m, color="#e74c3c", linewidth=1, alpha=0.8, label="qpos")
        ax.fill_between(iters, m - s, m + s, color="#e74c3c", alpha=0.08)
        # qvel
        m, s = qvel_means.mean(0), qvel_means.std(0)
        ax.plot(iters, m, color="#3498db", linewidth=1, alpha=0.8, label="qvel")
        # contact
        m, s = contact_means.mean(0), contact_means.std(0)
        ax.plot(iters, m, color="#2ecc71", linewidth=1, alpha=0.8, label="contact")

        if row == 0:
            ax.set_title(meta["label"], fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Tick {tick}\nReward", fontsize=10)
        if row == len(critical_ticks) - 1:
            ax.set_xlabel("Iteration")
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, n_iters - 1)

plt.suptitle("Per-Iteration Reward Breakdown at Critical Ticks (mean ± std, 3 seeds)", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_seeds_iteration_rewards.png"), dpi=150, bbox_inches="tight")
print("Saved fig2_seeds_iteration_rewards.png")

# ── Figure 3: Improvement heatmaps (averaged across seeds) ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for col, (cond, meta) in enumerate(CONDITIONS.items()):
    ax = axes[col]
    imp_stack = np.stack([d["improvement"] for d in all_data[cond]])  # (3, 17, 32)
    imp_mean = imp_stack.mean(axis=0)

    im = ax.imshow(imp_mean, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tick")
    ax.set_title(meta["label"])
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Improvement per Tick per Iteration (mean over 3 seeds)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_seeds_improvement_heatmap.png"), dpi=150, bbox_inches="tight")
print("Saved fig3_seeds_improvement_heatmap.png")

# ── Figure 4: Summary bar chart — final object z and tracking error ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cond_names = list(CONDITIONS.keys())
labels = [CONDITIONS[c]["label"] for c in cond_names]
colors = [CONDITIONS[c]["color"] for c in cond_names]
x = np.arange(len(cond_names))

# Final object z (last tick)
ax = axes[0]
final_z_means, final_z_stds = [], []
for cond in cond_names:
    vals = [get_obj_z(d["qpos"])[-1] for d in all_data[cond]]
    final_z_means.append(np.mean(vals))
    final_z_stds.append(np.std(vals))
ax.bar(x, final_z_means, yerr=final_z_stds, color=colors, capsize=5, alpha=0.8)
ax.axhline(y=ref_z_ticks[-1], color="black", linestyle="--", label="Reference")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Object z (m)")
ax.set_title("Final Object Height")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Mean tracking error across all ticks
ax = axes[1]
mean_err_means, mean_err_stds = [], []
for cond in cond_names:
    vals = [get_obj_pos_error(d["qpos"], n_ticks).mean() for d in all_data[cond]]
    mean_err_means.append(np.mean(vals))
    mean_err_stds.append(np.std(vals))
ax.bar(x, mean_err_means, yerr=mean_err_stds, color=colors, capsize=5, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean Position Error (m)")
ax.set_title("Average Object Tracking Error")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_seeds_summary_bar.png"), dpi=150)
print("Saved fig4_seeds_summary_bar.png")

print(f"\nDone! All figures saved to: {OUT}")
