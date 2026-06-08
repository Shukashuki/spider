"""
06_cross_rollout.py — Cross-rollout MPPI performance comparison for SPIDER.

Focuses on contact transition difficulty across rollouts for three bimanual tasks.

Data note: rew_mean has shape (N_ticks, 32) where axis-1 = MPPI iterations.
Rewards start negative at iter-0 and converge to ~0 by iter-31.
→ iter-0 captures *raw difficulty* before optimization.
→ mean across iters captures overall optimization cost.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path("example_datasets/processed/gigahand/xhand/bimanual")
OUT  = Path("notes/spider/viz")
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "p36-tea":        BASE / "p36-tea",
    "p44-dog":        BASE / "p44-dog",
    "p52-instrument": BASE / "p52-instrument",
}
N_ROLLOUTS = 5

# ── helpers ────────────────────────────────────────────────────────────────

def load_rollout(task_dir: Path, rollout_id: int):
    """Load a single rollout.

    Returns
    -------
    rew_init : 1-D (N_ticks,) — reward at MPPI iter 0 (pre-optimization).
    rew_avg  : 1-D (N_ticks,) — mean reward across all MPPI iterations.
    opt      : 1-D (N_ticks,) — optimizer steps used.
    """
    path = task_dir / str(rollout_id) / "trajectory_mjwp.npz"
    d = np.load(path)
    opt = d["opt_steps"].squeeze()

    if "rew_mean" in d:
        rew_init = d["rew_mean"][:, 0]
        rew_avg  = d["rew_mean"].mean(axis=1)
    elif "rew_sample" in d:
        summed = d["rew_sample"].sum(axis=-1)  # (N_ticks, N_iters)
        rew_init = summed[:, 0]
        rew_avg  = summed.mean(axis=1)
    else:
        raise KeyError(f"No reward key in {path}")

    return rew_init, rew_avg, opt


def load_task(task_name: str):
    task_dir = TASKS[task_name]
    rews_init, rews_avg, opts = [], [], []
    for r in range(N_ROLLOUTS):
        ri, ra, opt = load_rollout(task_dir, r)
        rews_init.append(ri)
        rews_avg.append(ra)
        opts.append(opt)
    return rews_init, rews_avg, opts


def common_length(arrays):
    return min(len(a) for a in arrays)


def stack_truncated(arrays):
    L = common_length(arrays)
    return np.stack([a[:L] for a in arrays], axis=0), L


# ── load everything ───────────────────────────────────────────────────────
data = {}
for task in TASKS:
    rews_init, rews_avg, opts = load_task(task)
    data[task] = {"rews_init": rews_init, "rews_avg": rews_avg, "opts": opts}
    lens = [len(r) for r in rews_init]
    print(f"{task}: rollout lengths = {lens}, common = {min(lens)}")

# ── Color palettes ────────────────────────────────────────────────────────
PALETTES = {
    "p36-tea":        plt.cm.Blues(np.linspace(0.35, 0.85, N_ROLLOUTS)),
    "p44-dog":        plt.cm.Oranges(np.linspace(0.35, 0.85, N_ROLLOUTS)),
    "p52-instrument": plt.cm.Greens(np.linspace(0.35, 0.85, N_ROLLOUTS)),
}
TASK_COLORS = {
    "p36-tea": "steelblue",
    "p44-dog": "darkorange",
    "p52-instrument": "seagreen",
}

# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Reward trends across rollouts (per task)
# ══════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig1.suptitle("Cross-Rollout Reward Trends (MPPI iter-0, raw difficulty)",
              fontsize=14, y=1.02)

outlier_report = {}

for ax, task in zip(axes1, TASKS):
    rews = data[task]["rews_init"]
    pal  = PALETTES[task]
    stacked, L = stack_truncated(rews)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    ticks = np.arange(L)

    for r in range(N_ROLLOUTS):
        full_t = np.arange(len(rews[r]))
        ax.plot(full_t, rews[r], color=pal[r], alpha=0.6, lw=1.2,
                label=f"rollout {r} (len={len(rews[r])})")

    ax.plot(ticks, mean, color="black", lw=2.5, label="mean", zorder=5)
    ax.fill_between(ticks, mean - std, mean + std, color="gray", alpha=0.15)

    # outliers: reward < mean - 1*std (more negative = harder)
    outliers = []
    for r in range(N_ROLLOUTS):
        for t in range(L):
            if stacked[r, t] < mean[t] - std[t]:
                ax.scatter(t, stacked[r, t], edgecolors="red", facecolors="none",
                           s=90, linewidths=1.8, zorder=6)
                outliers.append((r, t))
    outlier_report[task] = outliers

    ax.set_title(task, fontsize=12)
    ax.set_xlabel("Tick index")
    ax.set_ylabel("Reward (MPPI iter 0, pre-opt)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(OUT / "06a_reward_across_rollouts.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\n✓ Saved 06a_reward_across_rollouts.png")

# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Optimizer effort (opt_steps) across rollouts
# ══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig2.suptitle("Optimizer Effort Profile Across Rollouts", fontsize=14, y=1.02)

peak_report = {}

for ax, task in zip(axes2, TASKS):
    opts = data[task]["opts"]
    pal  = PALETTES[task]
    stacked_opt, L = stack_truncated(opts)
    mean_opt = stacked_opt.mean(axis=0)
    ticks = np.arange(L)

    for r in range(N_ROLLOUTS):
        full_t = np.arange(len(opts[r]))
        ax.plot(full_t, opts[r], color=pal[r], alpha=0.45, lw=1.5,
                label=f"rollout {r}")
        ax.bar(ticks + r * 0.15 - 0.3, stacked_opt[r], width=0.15,
               color=pal[r], alpha=0.3, edgecolor="none")

    ax.plot(ticks, mean_opt, color="black", lw=2.5, label="mean", zorder=5)

    peak_tick = int(np.argmax(mean_opt))
    peak_val  = mean_opt[peak_tick]
    ax.annotate(f"peak t={peak_tick}\n({peak_val:.0f} steps)",
                xy=(peak_tick, peak_val),
                xytext=(peak_tick + 1, peak_val * 1.05),
                fontsize=8, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                zorder=7)
    peak_report[task] = (peak_tick, peak_val)

    ax.set_title(task, fontsize=12)
    ax.set_xlabel("Tick index")
    ax.set_ylabel("opt_steps")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(OUT / "06b_effort_across_rollouts.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ Saved 06b_effort_across_rollouts.png")

# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Cross-task difficulty comparison
# ══════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3r = ax3.twinx()

difficulty_summary = {}

for task in TASKS:
    opts = data[task]["opts"]
    rews = data[task]["rews_init"]
    stacked_opt, Lo = stack_truncated(opts)
    stacked_rew, Lr = stack_truncated(rews)
    L = min(Lo, Lr)
    mean_opt = stacked_opt[:, :L].mean(axis=0)
    mean_rew = stacked_rew[:, :L].mean(axis=0)
    ticks = np.arange(L)
    c = TASK_COLORS[task]

    ax3.plot(ticks, mean_opt, color=c, lw=2.5, label=f"{task} opt_steps")
    ax3r.plot(ticks, mean_rew, color=c, lw=1.5, linestyle="--",
              label=f"{task} reward", alpha=0.7)

    difficulty_summary[task] = {
        "mean_opt": float(mean_opt.mean()),
        "mean_rew": float(mean_rew.mean()),
        "max_opt":  float(mean_opt.max()),
        "n_ticks":  L,
    }

ax3.set_xlabel("Tick index")
ax3.set_ylabel("opt_steps (mean across rollouts)", fontsize=10)
ax3r.set_ylabel("Reward at iter-0 (mean across rollouts)", fontsize=10)
ax3.set_title("Cross-Task Difficulty Comparison", fontsize=13)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(OUT / "06c_cross_task_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ Saved 06c_cross_task_comparison.png")

# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Reward variance across rollouts
# ══════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(12, 5))

variance_peaks = {}

for task in TASKS:
    rews = data[task]["rews_init"]
    stacked, L = stack_truncated(rews)
    var = stacked.var(axis=0)
    ticks = np.arange(L)
    c = TASK_COLORS[task]
    ax4.plot(ticks, var, color=c, lw=2, label=task)

    peak_t = int(np.argmax(var))
    peak_v = var[peak_t]
    ax4.annotate(f"t={peak_t} ({peak_v:.4f})",
                 xy=(peak_t, peak_v),
                 xytext=(peak_t + 0.8, peak_v * 1.08),
                 fontsize=8, color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    variance_peaks[task] = (peak_t, peak_v)

ax4.set_xlabel("Tick index")
ax4.set_ylabel("Reward variance across rollouts")
ax4.set_title("Reward Variability Analysis (contact instability indicator)", fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(OUT / "06d_reward_variance.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("✓ Saved 06d_reward_variance.png")

# ══════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

print("\n1. OUTLIER TICKS (reward < mean - 1 std, contact difficulty candidates):")
for task, outliers in outlier_report.items():
    if outliers:
        ticks_set = sorted(set(t for _, t in outliers))
        detail = ", ".join(f"r{r}@t{t}" for r, t in outliers)
        print(f"   {task}: {len(outliers)} outlier points at ticks {ticks_set}")
        print(f"      detail: {detail}")
    else:
        print(f"   {task}: no outliers detected")

print("\n2. DIFFICULTY RANKING (by mean opt_steps, higher = harder):")
ranked = sorted(difficulty_summary.items(), key=lambda x: -x[1]["mean_opt"])
for i, (task, s) in enumerate(ranked, 1):
    print(f"   #{i} {task}: mean_opt={s['mean_opt']:.1f}, "
          f"max_opt={s['max_opt']:.1f}, mean_rew={s['mean_rew']:.4f}, "
          f"n_ticks={s['n_ticks']}")

print(f"\n   HARDEST TASK: {ranked[0][0]} "
      f"(mean opt_steps = {ranked[0][1]['mean_opt']:.1f})")

print("\n3. PEAK OPT_STEPS TICKS (likely contact transitions):")
for task, (pt, pv) in peak_report.items():
    print(f"   {task}: peak at tick {pt} ({pv:.1f} steps)")

print("\n4. REWARD VARIANCE PEAKS (MPPI instability / contact):")
for task, (pt, pv) in variance_peaks.items():
    print(f"   {task}: peak variance at tick {pt} ({pv:.6f})")

print("\n" + "=" * 70)
print("All figures saved to notes/spider/viz/")
print("=" * 70)
