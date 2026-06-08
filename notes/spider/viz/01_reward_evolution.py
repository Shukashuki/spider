"""
SPIDER MPPI Reward Evolution Visualization
Analyzes reward convergence, improvement heatmap, and cross-tick reward trends.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------- paths ----------
REPO = Path("/home/roy/.openclaw/workspace/spider")
DATA = REPO / "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/trajectory_mjwp.npz"
OUT_DIR = REPO / "notes/spider/viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
data = np.load(DATA)
rew_max = data["rew_max"]       # (17, 32)
rew_mean = data["rew_mean"]     # (17, 32)
rew_min = data["rew_min"]       # (17, 32)
improvement = data["improvement"]  # (17, 32)
opt_steps = data["opt_steps"]   # (17, 1)

n_ticks, n_iters = rew_max.shape
opt_steps_flat = opt_steps.squeeze()  # (17,)

print(f"Loaded data: {n_ticks} ticks × {n_iters} iterations")
print(f"opt_steps per tick: {opt_steps_flat}")

# ============================================================
# Figure 1: Reward convergence per-tick (selected ticks)
# ============================================================
selected_ticks = [0, 5, 10, min(16, n_ticks - 1)]
fig1, axes = plt.subplots(1, len(selected_ticks), figsize=(5 * len(selected_ticks), 4), sharey=True)

for ax, tick in zip(axes, selected_ticks):
    steps = int(opt_steps_flat[tick])
    if steps == 0:
        steps = 1  # safety: at least show 1 point
    iters = np.arange(steps)

    ax.plot(iters, rew_max[tick, :steps], label="max", color="#e74c3c", linewidth=1.5)
    ax.plot(iters, rew_mean[tick, :steps], label="mean", color="#2980b9", linewidth=1.5)
    ax.plot(iters, rew_min[tick, :steps], label="min", color="#27ae60", linewidth=1.5)
    ax.fill_between(iters, rew_min[tick, :steps], rew_max[tick, :steps],
                     alpha=0.15, color="#2980b9")
    ax.set_title(f"Tick {tick}  (opt_steps={int(opt_steps_flat[tick])})", fontsize=11)
    ax.set_xlabel("Iteration")
    if ax is axes[0]:
        ax.set_ylabel("Reward")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

fig1.suptitle("MPPI Reward Convergence per Tick", fontsize=13, y=1.02)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "01a_reward_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("✓ 01a_reward_convergence.png saved")

# ============================================================
# Figure 2: Improvement heatmap
# ============================================================
masked_improvement = np.copy(improvement).astype(float)
for t in range(n_ticks):
    steps = int(opt_steps_flat[t])
    masked_improvement[t, steps:] = np.nan

fig2, ax2 = plt.subplots(figsize=(10, 5))
im = ax2.imshow(masked_improvement, aspect="auto", cmap="RdYlGn",
                origin="lower", interpolation="nearest")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Tick")
ax2.set_title("MPPI Improvement Heatmap (masked past opt_steps)", fontsize=13)
cbar = fig2.colorbar(im, ax=ax2, label="Improvement")

# draw opt_steps boundary
for t in range(n_ticks):
    steps = int(opt_steps_flat[t])
    if steps < n_iters:
        ax2.plot(steps - 0.5, t, "k|", markersize=8)

fig2.tight_layout()
fig2.savefig(OUT_DIR / "01b_improvement_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ 01b_improvement_heatmap.png saved")

# ============================================================
# Figure 3: Cross-tick reward trend (final iteration values)
# ============================================================
final_rew_max = np.array([rew_max[t, int(opt_steps_flat[t]) - 1] if opt_steps_flat[t] > 0 else rew_max[t, 0]
                          for t in range(n_ticks)])
final_rew_mean = np.array([rew_mean[t, int(opt_steps_flat[t]) - 1] if opt_steps_flat[t] > 0 else rew_mean[t, 0]
                           for t in range(n_ticks)])

fig3, ax3 = plt.subplots(figsize=(8, 4))
ticks_x = np.arange(n_ticks)
ax3.plot(ticks_x, final_rew_max, "o-", label="Final rew_max", color="#e74c3c", linewidth=1.5, markersize=5)
ax3.plot(ticks_x, final_rew_mean, "s-", label="Final rew_mean", color="#2980b9", linewidth=1.5, markersize=5)
ax3.fill_between(ticks_x, final_rew_mean, final_rew_max, alpha=0.12, color="#8e44ad")
ax3.set_xlabel("Tick Index")
ax3.set_ylabel("Reward")
ax3.set_title("Reward Trend Across Control Ticks (final iteration)", fontsize=13)
ax3.set_xticks(ticks_x)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(OUT_DIR / "01c_reward_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ 01c_reward_trend.png saved")

print("\nAll 3 figures saved to:", OUT_DIR)
