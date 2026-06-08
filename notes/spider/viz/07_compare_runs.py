"""
07 — Compare multiple MPPI runs (official baseline vs user experiments)
Generates 4-panel figure:
  1. Reward (mean ± min/max) over ticks
  2. qpos distance (mean) over ticks
  3. Optimization steps per tick
  4. Cost breakdown (trace_cost) stacked bars
"""
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0"

runs = {
    "Baseline (official)": os.path.join(BASE, "trajectory_mjwp_baseline.npz"),
    "Exp2 (T=0.3, th=0.001)": os.path.join(BASE, "trajectory_mjwp_t001.npz"),
    "Exp3 (T=0.1, th=0.001)": os.path.join(BASE, "trajectory_mjwp.npz"),
}

colors = {
    "Baseline (official)": "#2196F3",
    "Exp2 (T=0.3, th=0.001)": "#FF9800",
    "Exp3 (T=0.1, th=0.001)": "#4CAF50",
}

data = {}
for label, path in runs.items():
    d = np.load(path, allow_pickle=True)
    data[label] = {k: d[k] for k in d.keys()}
    print(f"Loaded {label}: {path}")

n_ticks = data["Baseline (official)"]["rew_mean"].shape[0]
ticks = np.arange(n_ticks)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SPIDER p36-tea: MPPI Run Comparison (xhand, rollout 0)", fontsize=14, fontweight="bold")

# --- Panel 1: Reward over ticks ---
ax = axes[0, 0]
for label, d in data.items():
    # rew_mean shape: (17, 32) — take the last iteration's value as "final" reward
    rew_final = d["rew_mean"][:, -1]  # last iteration per tick
    rew_min = d["rew_min"][:, -1]
    rew_max = d["rew_max"][:, -1]
    ax.plot(ticks, rew_final, "o-", color=colors[label], label=label, markersize=4)
    ax.fill_between(ticks, rew_min, rew_max, alpha=0.15, color=colors[label])
ax.set_xlabel("Tick")
ax.set_ylabel("Reward (final iter)")
ax.set_title("1. Reward per Tick (mean ± min/max)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: qpos distance over ticks ---
ax = axes[0, 1]
for label, d in data.items():
    qd_final = d["qpos_dist_mean"][:, -1]
    qd_min = d["qpos_dist_min"][:, -1]
    qd_max = d["qpos_dist_max"][:, -1]
    ax.plot(ticks, qd_final, "s-", color=colors[label], label=label, markersize=4)
    ax.fill_between(ticks, qd_min, qd_max, alpha=0.15, color=colors[label])
ax.set_xlabel("Tick")
ax.set_ylabel("qpos distance (final iter)")
ax.set_title("2. qpos Distance per Tick")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: Optimization steps per tick ---
ax = axes[1, 0]
bar_width = 0.25
offsets = np.array([-1, 0, 1]) * bar_width
for i, (label, d) in enumerate(data.items()):
    steps = d["opt_steps"][:, 0]
    ax.bar(ticks + offsets[i], steps, bar_width, label=label, color=colors[label], alpha=0.8)
ax.set_xlabel("Tick")
ax.set_ylabel("opt_steps")
ax.set_title("3. Optimization Steps per Tick")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 4: Reward convergence (iteration trajectory, averaged over ticks) ---
ax = axes[1, 1]
for label, d in data.items():
    # rew_mean: (17, 32) — average across ticks to get convergence curve
    rew_curve = d["rew_mean"].mean(axis=0)  # (32,)
    n_iter = len(rew_curve)
    ax.plot(np.arange(n_iter), rew_curve, "-", color=colors[label], label=label, linewidth=2)
ax.set_xlabel("MPPI Iteration")
ax.set_ylabel("Mean Reward (avg over ticks)")
ax.set_title("4. Reward Convergence (avg across ticks)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "notes/spider/viz/07_compare_runs.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")

# --- Summary stats ---
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for label, d in data.items():
    rew = d["rew_mean"][:, -1]
    qd = d["qpos_dist_mean"][:, -1]
    steps = d["opt_steps"][:, 0]
    print(f"\n{label}:")
    print(f"  reward (final iter):  mean={rew.mean():.4f}  min={rew.min():.4f}  max={rew.max():.4f}")
    print(f"  qpos_dist (final):    mean={qd.mean():.4f}  min={qd.min():.4f}  max={qd.max():.4f}")
    print(f"  opt_steps:            mean={steps.mean():.1f}  min={steps.min()}  max={steps.max()}")

# --- Bonus: trace_cost breakdown ---
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Cost Breakdown by Component (trace_cost, final iter)", fontsize=13, fontweight="bold")
cost_labels = [f"c{i}" for i in range(6)]

for idx, (label, d) in enumerate(data.items()):
    ax = axes2[idx]
    # trace_cost: (17, 32, 6) — take last iteration
    tc = d["trace_cost"][:, -1, :]  # (17, 6)
    bottom = np.zeros(n_ticks)
    cmap = plt.cm.Set2
    for ci in range(6):
        ax.bar(ticks, tc[:, ci], bottom=bottom, label=cost_labels[ci],
               color=cmap(ci / 6), alpha=0.85, width=0.8)
        bottom += tc[:, ci]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Cost")
    ax.set_title(label, fontsize=10)
    if idx == 0:
        ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path2 = "notes/spider/viz/07_cost_breakdown.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path2}")
