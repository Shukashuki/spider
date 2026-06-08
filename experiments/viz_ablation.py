"""Visualize ablation results: per-tick, per-iteration reward breakdown.

Generates:
1. Per-tick object tracking error comparison across conditions
2. Per-iteration reward curves for worst ticks
3. Reward component breakdown (qpos_rew, qvel_rew, contact_rew)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "ablation_contact", "p36-tea")

CONDITIONS = {
    "A_original": {"suffix": "_act", "label": "A: gain decay", "color": "#e74c3c"},
    "B_no_gain_decay": {"suffix": "", "label": "B: nothing", "color": "#3498db"},
    "C_gain_decay_plus_cost": {"suffix": "_act", "label": "C: decay+cost", "color": "#2ecc71"},
    "D_cost_only": {"suffix": "", "label": "D: cost only", "color": "#f39c12"},
}

# Load config
with open(os.path.join(OUT, "A_original", "config_act.yaml")) as f:
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

# Load all data
data = {}
for cond, meta in CONDITIONS.items():
    path = os.path.join(OUT, cond, f"trajectory_mjwp{meta['suffix']}.npz")
    data[cond] = dict(np.load(path))

n_ticks = data["A_original"]["qpos"].shape[0]
n_iters = data["A_original"]["rew_mean"].shape[1]

# ── Figure 1: Object z trajectory + tracking error ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Reference obj_z (sampled at tick boundaries)
ref_z_ticks = []
for t in range(n_ticks):
    ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
    ref_z_ticks.append(ref_qpos[ref_step, obj_z_idx])
ref_z_ticks = np.array(ref_z_ticks)

ax = axes[0]
ax.plot(range(n_ticks), ref_z_ticks, "k--", linewidth=2, label="Reference", zorder=10)
for cond, meta in CONDITIONS.items():
    qpos = data[cond]["qpos"]
    obj_z = qpos[:, -1, obj_z_idx]
    ax.plot(range(n_ticks), obj_z, "-o", color=meta["color"], label=meta["label"], markersize=4)
ax.set_ylabel("Object z (m)")
ax.set_title("Object Height Trajectory")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Tracking error
ax = axes[1]
for cond, meta in CONDITIONS.items():
    qpos = data[cond]["qpos"]
    errs = []
    for t in range(n_ticks):
        ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
        sim_obj = qpos[t, -1, nq - nq_obj:nq - nq_obj + 3]
        ref_obj = ref_qpos[ref_step, nq - nq_obj:nq - nq_obj + 3]
        errs.append(np.linalg.norm(sim_obj - ref_obj))
    ax.plot(range(n_ticks), errs, "-o", color=meta["color"], label=meta["label"], markersize=4)
ax.set_ylabel("Object Position Error (m)")
ax.set_xlabel("Tick")
ax.set_title("Object Tracking Error per Tick")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_obj_trajectory.png"), dpi=150)
print(f"Saved fig1_obj_trajectory.png")

# ── Figure 2: Per-iteration reward curves for worst ticks ──
# Find worst ticks per condition
worst_ticks_per_cond = {}
for cond, meta in CONDITIONS.items():
    qpos = data[cond]["qpos"]
    errs = []
    for t in range(n_ticks):
        ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
        sim_obj = qpos[t, -1, nq - nq_obj:nq - nq_obj + 3]
        ref_obj = ref_qpos[ref_step, nq - nq_obj:nq - nq_obj + 3]
        errs.append(np.linalg.norm(sim_obj - ref_obj))
    worst_ticks_per_cond[cond] = np.argsort(errs)[::-1][:3]

# Global worst ticks (union of top-3 from each condition)
all_worst = set()
for ticks in worst_ticks_per_cond.values():
    all_worst.update(ticks.tolist())
# Also add tick 4 (where object should be lifting) and tick 10
all_worst.update([4, 10])
worst_ticks = sorted(all_worst)[:6]

fig, axes = plt.subplots(len(worst_ticks), 4, figsize=(20, 3.5 * len(worst_ticks)))
if len(worst_ticks) == 1:
    axes = axes[np.newaxis, :]

for row, tick in enumerate(worst_ticks):
    for col, (cond, meta) in enumerate(CONDITIONS.items()):
        ax = axes[row, col]
        d = data[cond]
        opt_steps_actual = int(d["opt_steps"][tick, 0])

        iters = range(n_iters)
        ax.plot(iters, d["rew_mean"][tick], color="black", linewidth=1.5, label="total")
        ax.plot(iters, d["qpos_rew_mean"][tick], color="#e74c3c", linewidth=1, alpha=0.8, label="qpos")
        ax.plot(iters, d["qvel_rew_mean"][tick], color="#3498db", linewidth=1, alpha=0.8, label="qvel")
        ax.plot(iters, d["contact_rew_mean"][tick], color="#2ecc71", linewidth=1, alpha=0.8, label="contact")

        # Mark actual opt steps
        ax.axvline(x=opt_steps_actual - 1, color="gray", linestyle=":", alpha=0.5)

        # Shade min-max
        ax.fill_between(iters, d["rew_min"][tick], d["rew_max"][tick], alpha=0.1, color="black")

        if row == 0:
            ax.set_title(f"{meta['label']}", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Tick {tick}\nReward", fontsize=10)
        if row == len(worst_ticks) - 1:
            ax.set_xlabel("Iteration")
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, n_iters - 1)

plt.suptitle("Per-Iteration Reward Breakdown at Critical Ticks", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_iteration_rewards.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig2_iteration_rewards.png")

# ── Figure 3: Improvement curves ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for col, (cond, meta) in enumerate(CONDITIONS.items()):
    ax = axes[col]
    d = data[cond]
    imp = d["improvement"]  # (17, 32)

    # Heatmap: ticks x iterations
    im = ax.imshow(imp, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tick")
    ax.set_title(meta["label"])
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Improvement per Tick per Iteration", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_improvement_heatmap.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig3_improvement_heatmap.png")

# ── Figure 4: Reward distribution (min/max/mean) across iterations for key ticks ──
key_ticks = [0, 4, 8, 12, 16]
fig, axes = plt.subplots(len(key_ticks), 4, figsize=(20, 3 * len(key_ticks)))

for row, tick in enumerate(key_ticks):
    for col, (cond, meta) in enumerate(CONDITIONS.items()):
        ax = axes[row, col]
        d = data[cond]

        ax.fill_between(range(n_iters), d["rew_min"][tick], d["rew_max"][tick],
                        alpha=0.2, color=meta["color"], label="min-max")
        ax.plot(range(n_iters), d["rew_mean"][tick], color=meta["color"],
                linewidth=1.5, label="mean")
        ax.plot(range(n_iters), d["rew_median"][tick], color=meta["color"],
                linewidth=1, linestyle="--", alpha=0.7, label="median")

        if row == 0:
            ax.set_title(meta["label"], fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Tick {tick}\nReward")
        if row == len(key_ticks) - 1:
            ax.set_xlabel("Iteration")
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

plt.suptitle("Reward Distribution (min/mean/median/max) Across Iterations", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_reward_distribution.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig4_reward_distribution.png")

print("\nDone! All figures saved to:", OUT)
