"""Visualize MPPI vs MPPI-CMA on p36-tea (from quick dryrun agg data).

Labels: MPPI → "Spider-Origin-Optimizer", Gain Decay → "Contact Guidance".

Usage:
    cd spider/
    .venv/bin/python experiments/viz_optimizer_tea.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "mppi_cma_quick_dryrun_tea")

EXPS = {
    "gain_decay__mppi": {"label": "Contact Guidance + Spider-Origin-Optimizer", "color": "#e74c3c", "ls": "-"},
    "gain_decay__mppi_cma": {"label": "Contact Guidance + MPPI-CMA", "color": "#e74c3c", "ls": "--"},
    "no_gain_decay__mppi": {"label": "No Contact Guidance + Spider-Origin-Optimizer", "color": "#3498db", "ls": "-"},
    "no_gain_decay__mppi_cma": {"label": "No Contact Guidance + MPPI-CMA", "color": "#3498db", "ls": "--"},
}

# Load all agg data
data = {}
for key in EXPS:
    data[key] = dict(np.load(os.path.join(OUT, f"{key}_agg.npz")))

sample = data["gain_decay__mppi"]
tick_ids = sample["tick_ids"]
n_seeds = sample["rew_mean"].shape[0]
n_ticks = sample["rew_mean"].shape[1]
n_iters = sample["rew_mean"].shape[2]
print(f"tick_ids={tick_ids}, n_seeds={n_seeds}, n_ticks={n_ticks}, n_iters={n_iters}")

# ── Fig 1: All 4 combos convergence per tick ──
fig, axes = plt.subplots(1, n_ticks, figsize=(6 * n_ticks, 5), sharey=True)
if n_ticks == 1:
    axes = [axes]

for t_idx in range(n_ticks):
    ax = axes[t_idx]
    tick = int(tick_ids[t_idx])
    for key, meta in EXPS.items():
        rew = data[key]["rew_mean"][:, t_idx, :]  # (seeds, iters)
        m, s = rew.mean(0), rew.std(0)
        ax.plot(range(n_iters), m, color=meta["color"], linestyle=meta["ls"],
                linewidth=2, label=meta["label"])
        ax.fill_between(range(n_iters), m - s, m + s, color=meta["color"], alpha=0.1)
    ax.set_title(f"Tick {tick}", fontsize=12)
    ax.set_xlabel("Iteration")
    if t_idx == 0:
        ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    if t_idx == 0:
        ax.legend(fontsize=6, loc="lower right")

plt.suptitle("p36-tea: Reward Convergence (3 seeds, ctrl_dt=0.1)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_convergence_all.png"), dpi=150, bbox_inches="tight")
print("Saved fig1_convergence_all.png")

# ── Fig 2: Optimizer comparison (Contact Guidance only) ──
fig, axes = plt.subplots(1, n_ticks, figsize=(6 * n_ticks, 5), sharey=True)
if n_ticks == 1:
    axes = [axes]

OPT_COMPARE = {
    "gain_decay__mppi": {"label": "Spider-Origin-Optimizer", "color": "#e74c3c"},
    "gain_decay__mppi_cma": {"label": "MPPI-CMA", "color": "#3498db"},
}

for t_idx in range(n_ticks):
    ax = axes[t_idx]
    tick = int(tick_ids[t_idx])
    for key, meta in OPT_COMPARE.items():
        rew = data[key]["rew_mean"][:, t_idx, :]
        m, s = rew.mean(0), rew.std(0)
        ax.plot(range(n_iters), m, color=meta["color"], linewidth=2, label=meta["label"])
        ax.fill_between(range(n_iters), m - s, m + s, color=meta["color"], alpha=0.15)
    ax.set_title(f"Tick {tick}", fontsize=12)
    ax.set_xlabel("Iteration")
    if t_idx == 0:
        ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    if t_idx == 0:
        ax.legend(fontsize=9)

plt.suptitle("p36-tea Contact Guidance: Spider-Origin-Optimizer vs MPPI-CMA (3 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_optimizer_compare.png"), dpi=150, bbox_inches="tight")
print("Saved fig2_optimizer_compare.png")

# ── Fig 3: Reward components at tick 5 (2×2 grid) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
GRID = [
    [("gain_decay__mppi", "Contact Guidance\nSpider-Origin-Optimizer"),
     ("no_gain_decay__mppi", "No Contact Guidance\nSpider-Origin-Optimizer")],
    [("gain_decay__mppi_cma", "Contact Guidance\nMPPI-CMA"),
     ("no_gain_decay__mppi_cma", "No Contact Guidance\nMPPI-CMA")],
]

target_tick = 5
t_idx = np.where(tick_ids == target_tick)[0]
t_idx = t_idx[0] if len(t_idx) > 0 else 0
actual_tick = int(tick_ids[t_idx])

for row in range(2):
    for col in range(2):
        ax = axes[row, col]
        key, title = GRID[row][col]
        d = data[key]
        iters = range(n_iters)

        for comp, color, lbl in [
            ("qpos_rew_mean", "#e74c3c", "qpos"),
            ("qvel_rew_mean", "#3498db", "qvel"),
            ("contact_rew_mean", "#2ecc71", "contact"),
        ]:
            curve_all = d[comp][:, t_idx, :]  # (seeds, iters)
            m = curve_all.mean(0)
            s = curve_all.std(0)
            ax.plot(iters, m, color=color, label=lbl, linewidth=1.2)
            ax.fill_between(iters, m - s, m + s, color=color, alpha=0.1)

        total_all = d["rew_mean"][:, t_idx, :]
        m = total_all.mean(0)
        s = total_all.std(0)
        ax.plot(iters, m, "k-", linewidth=1.5, label="total")
        ax.fill_between(iters, m - s, m + s, color="black", alpha=0.08)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

plt.suptitle(f"p36-tea: Reward Components at Tick {actual_tick} (mean ± std, 3 seeds)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_components_tick5.png"), dpi=150, bbox_inches="tight")
print("Saved fig3_components_tick5.png")

# ── Fig 4: Convergence speed bar chart ──
fig, ax = plt.subplots(figsize=(10, 5))

bar_data = {}
for key, meta in EXPS.items():
    conv_iters = []
    d = data[key]
    for s in range(n_seeds):
        for t in range(n_ticks):
            rew = d["rew_mean"][s, t, :]
            final, init = rew[-1], rew[0]
            if abs(final - init) < 1e-8:
                conv_iters.append(0)
            else:
                threshold = init + 0.9 * (final - init)
                if final < init:
                    crossed = np.where(rew <= threshold)[0]
                else:
                    crossed = np.where(rew >= threshold)[0]
                conv_iters.append(crossed[0] if len(crossed) > 0 else n_iters)
    bar_data[key] = conv_iters

x = np.arange(len(EXPS))
means = [np.mean(bar_data[k]) for k in EXPS]
stds = [np.std(bar_data[k]) for k in EXPS]
colors = [EXPS[k]["color"] for k in EXPS]
hatches = [None, "//", None, "//"]

bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.8, edgecolor="black")
for i, h in enumerate(hatches):
    if h:
        bars[i].set_hatch(h)

ax.set_xticks(x)
ax.set_xticklabels([EXPS[k]["label"] for k in EXPS], fontsize=7, rotation=15, ha="right")
ax.set_ylabel("Iterations to 90% of Final Reward")
ax.set_title("p36-tea: Convergence Speed Comparison (3 seeds)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_convergence_speed.png"), dpi=150)
print("Saved fig4_convergence_speed.png")

print(f"\nDone! All figures saved to: {OUT}")
