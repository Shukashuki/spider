"""
SPIDER Dashboard — Full Metrics Visualization
Uses core/loader.py + core/analyzer.py to produce a 6-panel summary figure.
"""
import sys
sys.path.insert(0, "/home/roy/.openclaw/workspace/spider/dashboard")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from core.loader import load_trajectory
from core.analyzer import compute_metrics, compute_tick_boundary_discontinuity

# ---------- Config ----------
REPO = Path("/home/roy/.openclaw/workspace/spider")
TASKS = {
    "p36-tea": REPO / "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0",
    "p44-dog": REPO / "example_datasets/processed/gigahand/xhand/bimanual/p44-dog/0",
    "p52-instrument": REPO / "example_datasets/processed/gigahand/xhand/bimanual/p52-instrument/0",
}
OUT_DIR = REPO / "dashboard" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Color palette ----------
COLORS = {
    "p36-tea": "#e74c3c",
    "p44-dog": "#2980b9",
    "p52-instrument": "#27ae60",
}

# ---------- Load all ----------
print("Loading trajectories...")
all_data = {}
all_metrics = {}
for name, path in TASKS.items():
    try:
        data = load_trajectory(path)
        metrics = compute_metrics(data)
        all_data[name] = data
        all_metrics[name] = metrics
        print(f"  ✓ {name}: {data.meta['n_ticks']} ticks, "
              f"opt_steps={metrics.opt_steps_mean:.1f}, "
              f"reward={metrics.reward_mean:.4f}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# ============================================================
# Figure 1: 6-Panel Dashboard Summary
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# --- Panel 1: Reward Trend (per tick) ---
ax1 = fig.add_subplot(gs[0, 0])
for name, m in all_metrics.items():
    ax1.plot(m.reward_trend, "-", color=COLORS[name], label=name,
             linewidth=1.5, alpha=0.85)
ax1.set_xlabel("Tick")
ax1.set_ylabel("Final Reward")
ax1.set_title("Reward Trend (per tick)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Opt Steps (per tick) ---
ax2 = fig.add_subplot(gs[0, 1])
for name, m in all_metrics.items():
    ax2.plot(m.opt_steps_per_tick, "o-", color=COLORS[name], label=name,
             linewidth=1.2, markersize=3, alpha=0.85)
ax2.set_xlabel("Tick")
ax2.set_ylabel("Convergence Steps")
ax2.set_title("Optimizer Effort (per tick)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Tick Boundary Discontinuity (qpos) ---
ax3 = fig.add_subplot(gs[0, 2])
for name, d in all_data.items():
    qpos = d.qpos  # (n_ticks, steps, nq)
    last = qpos[:-1, -1, :]
    first = qpos[1:, 0, :]
    jumps = np.linalg.norm(last - first, axis=-1)
    ax3.plot(jumps, "-", color=COLORS[name], label=name, linewidth=1.5, alpha=0.85)
ax3.set_xlabel("Tick Boundary")
ax3.set_ylabel("L2 Jump (rad)")
ax3.set_title("qpos Discontinuity at Tick Boundaries", fontsize=12, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Cost Breakdown (bar chart) ---
ax4 = fig.add_subplot(gs[1, 0])
task_names = list(all_metrics.keys())
n_costs = max(len(m.cost_breakdown) for m in all_metrics.values())
x = np.arange(n_costs)
width = 0.25
for i, name in enumerate(task_names):
    m = all_metrics[name]
    vals = [m.cost_breakdown.get(f"cost_{j}", 0) for j in range(n_costs)]
    ax4.bar(x + i * width, vals, width, label=name, color=COLORS[name], alpha=0.8)
ax4.set_xlabel("Cost Component")
ax4.set_ylabel("Mean Cost")
ax4.set_title("Cost Breakdown by Component", fontsize=12, fontweight="bold")
ax4.set_xticks(x + width)
ax4.set_xticklabels([f"c{i}" for i in range(n_costs)], fontsize=8)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

# --- Panel 5: Summary Table ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis("off")
col_labels = ["Task", "opt_μ", "opt_σ", "rew_μ", "qpos_disc", "ctrl_disc", "rew_var"]
table_data = []
for name, m in all_metrics.items():
    table_data.append([
        name,
        f"{m.opt_steps_mean:.1f}",
        f"{m.opt_steps_std:.1f}",
        f"{m.reward_mean:.4f}",
        f"{m.qpos_discontinuity_mean:.4f}",
        f"{m.ctrl_discontinuity_mean:.4f}",
        f"{m.reward_variance:.4f}",
    ])
table = ax5.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
# color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")
# color rows
for i, name in enumerate(task_names):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(COLORS[name] + "18")
ax5.set_title("Experiment Summary", fontsize=12, fontweight="bold", pad=20)

# --- Panel 6: Reward Convergence (selected tick, p36-tea) ---
ax6 = fig.add_subplot(gs[1, 2])
tea_data = all_data.get("p36-tea")
if tea_data is not None:
    rew_mean = tea_data.rew_mean
    rew_min = tea_data.rew_min
    rew_max = tea_data.rew_max
    opt = tea_data.opt_steps.squeeze()
    # pick tick with max opt_steps (hardest tick)
    hardest = int(np.argmax(opt))
    steps = int(opt[hardest])
    iters = np.arange(steps)
    ax6.plot(iters, rew_max[hardest, :steps], label="max", color="#e74c3c", linewidth=1.5)
    ax6.plot(iters, rew_mean[hardest, :steps], label="mean", color="#2980b9", linewidth=1.5)
    ax6.plot(iters, rew_min[hardest, :steps], label="min", color="#27ae60", linewidth=1.5)
    ax6.fill_between(iters, rew_min[hardest, :steps], rew_max[hardest, :steps],
                     alpha=0.12, color="#8e44ad")
    ax6.set_xlabel("MPPI Iteration")
    ax6.set_ylabel("Reward")
    ax6.set_title(f"Reward Convergence — p36-tea Tick {hardest} (hardest)", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

fig.suptitle("SPIDER MPPI Dashboard — 3-Task Comparison (gigahand/xhand)", 
             fontsize=15, fontweight="bold", y=0.98)
fig.savefig(OUT_DIR / "dashboard_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ Dashboard saved to {OUT_DIR / 'dashboard_summary.png'}")

# ============================================================
# Figure 2: Reward Evolution Heatmap (p36-tea)
# ============================================================
if tea_data is not None:
    fig2, ax = plt.subplots(figsize=(12, 5))
    # Build a proper heatmap: mask beyond opt_steps
    rew = tea_data.rew_mean.copy().astype(float)
    opt_flat = tea_data.opt_steps.squeeze()
    for t in range(rew.shape[0]):
        s = int(opt_flat[t])
        rew[t, s:] = np.nan
    
    im = ax.imshow(rew, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
    ax.set_xlabel("MPPI Iteration")
    ax.set_ylabel("Tick")
    ax.set_title("Reward Evolution Heatmap — p36-tea (masked beyond convergence)", fontsize=13, fontweight="bold")
    fig2.colorbar(im, ax=ax, label="Reward (rew_mean)")
    
    # mark convergence boundary
    for t in range(len(opt_flat)):
        s = int(opt_flat[t])
        if s < rew.shape[1]:
            ax.plot(s - 0.5, t, "w|", markersize=10, markeredgewidth=2)
    
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "reward_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Reward heatmap saved to {OUT_DIR / 'reward_heatmap.png'}")

print("\nDone!")
