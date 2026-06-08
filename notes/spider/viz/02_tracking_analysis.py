"""
02_tracking_analysis.py
Visualize SPIDER tracking reward components and optimizer effort.

Outputs:
  02a_tracking_reward.png    — qpos/qvel reward over ticks
  02b_reward_components.png  — qpos vs qvel reward ratio
  02c_optimizer_effort.png   — opt_steps + improvement per tick
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0")
OUT_DIR = Path("notes/spider/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────
d = np.load(DATA_DIR / "trajectory_mjwp.npz")
opt_steps = d["opt_steps"].flatten()          # (17,)
improvement = d["improvement"]                # (17, 32)
n_ticks = len(opt_steps)
ticks = np.arange(n_ticks)

# Helper: grab the value at the *last actual iteration* for each tick
def last_iter(arr):
    """arr shape (17, 32) → (17,) picking index opt_steps[t]-1 per tick."""
    return np.array([arr[t, int(opt_steps[t]) - 1] for t in range(n_ticks)])

qpos_rew_max  = last_iter(d["qpos_rew_max"])
qpos_rew_mean = last_iter(d["qpos_rew_mean"])
qpos_rew_min  = last_iter(d["qpos_rew_min"])

qvel_rew_max  = last_iter(d["qvel_rew_max"])
qvel_rew_mean = last_iter(d["qvel_rew_mean"])
qvel_rew_min  = last_iter(d["qvel_rew_min"])

imp_last = last_iter(improvement)

# ══════════════════════════════════════════════════════════════════════
# Figure 1: qpos & qvel reward components over ticks
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ── qpos subplot ──
ax1.fill_between(ticks, qpos_rew_min, qpos_rew_max, alpha=0.2, color="tab:blue",
                 label="min–max range")
ax1.plot(ticks, qpos_rew_mean, "o-", color="tab:blue", lw=2, label="mean")
ax1.plot(ticks, qpos_rew_max,  "^--", color="tab:cyan", ms=5, lw=1, label="max")
ax1.plot(ticks, qpos_rew_min,  "v--", color="tab:purple", ms=5, lw=1, label="min")
ax1.set_ylabel("qpos reward (last iter)")
ax1.set_title("Tracking Reward Components per Tick (last optimization iteration)")
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color="grey", lw=0.5)

# ── qvel subplot ──
ax2.fill_between(ticks, qvel_rew_min, qvel_rew_max, alpha=0.2, color="tab:orange",
                 label="min–max range")
ax2.plot(ticks, qvel_rew_mean, "o-", color="tab:orange", lw=2, label="mean")
ax2.plot(ticks, qvel_rew_max,  "^--", color="tab:red", ms=5, lw=1, label="max")
ax2.plot(ticks, qvel_rew_min,  "v--", color="tab:brown", ms=5, lw=1, label="min")
ax2.set_xlabel("Tick index")
ax2.set_ylabel("qvel reward (last iter)")
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color="grey", lw=0.5)
ax2.set_xticks(ticks)

fig.tight_layout()
fig.savefig(OUT_DIR / "02a_tracking_reward.png", dpi=150)
plt.close(fig)
print("✓ 02a_tracking_reward.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Reward component ratio  (qpos vs qvel)
# ══════════════════════════════════════════════════════════════════════
# Use absolute values for ratio (rewards are negative)
abs_qpos = np.abs(qpos_rew_mean)
abs_qvel = np.abs(qvel_rew_mean)
total = abs_qpos + abs_qvel
frac_qpos = abs_qpos / total * 100
frac_qvel = abs_qvel / total * 100

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7),
                                      gridspec_kw={"height_ratios": [2, 1]})

# ── stacked area ──
ax_top.fill_between(ticks, 0, frac_qpos, alpha=0.6, color="tab:blue",
                    label="qpos (position)")
ax_top.fill_between(ticks, frac_qpos, 100, alpha=0.6, color="tab:orange",
                    label="qvel (velocity)")
ax_top.set_ylabel("Cost share (%)")
ax_top.set_ylim(0, 100)
ax_top.set_title("Reward Cost Breakdown: qpos vs qvel")
ax_top.legend(loc="center right", fontsize=10)
ax_top.grid(True, alpha=0.3, axis="y")
ax_top.set_xticks(ticks)

# ── absolute values (log scale) ──
ax_bot.semilogy(ticks, abs_qpos, "s-", color="tab:blue", lw=2, label="|qpos_rew_mean|")
ax_bot.semilogy(ticks, abs_qvel, "D-", color="tab:orange", lw=2, label="|qvel_rew_mean|")
ax_bot.set_xlabel("Tick index")
ax_bot.set_ylabel("Absolute reward (log)")
ax_bot.legend(fontsize=9)
ax_bot.grid(True, alpha=0.3, which="both")
ax_bot.set_xticks(ticks)

fig.tight_layout()
fig.savefig(OUT_DIR / "02b_reward_components.png", dpi=150)
plt.close(fig)
print("✓ 02b_reward_components.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Optimizer effort
# ══════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(12, 5))

# ── opt_steps bar ──
colors = ["tab:red" if s >= 20 else "tab:orange" if s >= 10 else "tab:green"
          for s in opt_steps]
bars = ax1.bar(ticks, opt_steps, color=colors, alpha=0.7, label="opt_steps")
ax1.set_ylabel("Optimization steps", color="tab:blue")
ax1.set_xlabel("Tick index")
ax1.set_title("Optimizer Effort per Tick")
ax1.axhline(32, color="grey", ls="--", lw=1, label="max iters (32)")
ax1.set_xticks(ticks)
ax1.set_ylim(0, 35)

# annotate step counts on bars
for t, s in enumerate(opt_steps):
    ax1.text(t, s + 0.5, str(s), ha="center", va="bottom", fontsize=8,
             fontweight="bold")

# ── improvement on twin axis ──
ax2 = ax1.twinx()
ax2.plot(ticks, imp_last, "ko-", lw=2, ms=6, label="improvement (last iter)")
ax2.set_ylabel("Improvement (last iter)", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# highlight "struggling" ticks: high steps OR low improvement
struggle_mask = (opt_steps >= 15) | (imp_last < 0.003)
for t in ticks[struggle_mask]:
    ax1.axvspan(t - 0.4, t + 0.4, alpha=0.1, color="red")

# combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

ax1.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "02c_optimizer_effort.png", dpi=150)
plt.close(fig)
print("✓ 02c_optimizer_effort.png")

print("\nDone — all plots saved to", OUT_DIR.resolve())
