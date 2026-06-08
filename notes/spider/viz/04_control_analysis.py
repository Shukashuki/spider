"""
04_control_analysis.py — Analyse SPIDER control signals (ctrl).

Generates three figures:
  04a_ctrl_timeseries.png   — Selected actuator ctrl over flattened time
  04b_ctrl_discontinuity.png — Heatmap of |jump| at tick boundaries
  04c_ctrl_effort.png       — Per-tick control effort + opt_steps overlay
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parents[3] / (
    "example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/trajectory_mjwp.npz"
)
OUT_DIR = Path(__file__).resolve().parent
assert DATA.exists(), f"Data not found: {DATA}"

data = np.load(DATA)
ctrl = data["ctrl"]          # (17, 40, 36)
time = data["time"]          # (17, 40)
opt_steps = data["opt_steps"].flatten()  # (17,)

n_ticks, n_steps, n_act = ctrl.shape
print(f"ctrl shape: {ctrl.shape}  |  time range: {time.min():.3f}–{time.max():.3f}s")

# ── actuator labels (bimanual: 0-17 left, 18-35 right, best guess) ────
SELECTED = {
    0:  "L-wrist-0",
    3:  "L-index-MCP",
    6:  "L-middle-MCP",
    9:  "L-ring-MCP",
    18: "R-wrist-0",
    21: "R-index-MCP",
    24: "R-middle-MCP",
    27: "R-ring-MCP",
}

# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Control signal time-series
# ═══════════════════════════════════════════════════════════════════════
time_flat = time.reshape(-1)                      # (680,)
ctrl_flat = ctrl.reshape(n_ticks * n_steps, n_act) # (680, 36)

fig1, ax1 = plt.subplots(figsize=(14, 6))
cmap = plt.cm.tab10
for i, (idx, label) in enumerate(SELECTED.items()):
    ax1.plot(time_flat, ctrl_flat[:, idx], lw=0.9, label=label, color=cmap(i))

# tick boundaries (vertical dashed lines)
for t in range(1, n_ticks):
    boundary_time = time[t, 0]
    ax1.axvline(boundary_time, color="grey", ls="--", lw=0.5, alpha=0.6)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("ctrl value")
ax1.set_title("Control Signals — Selected Actuators over Time")
ax1.legend(fontsize=7, ncol=2, loc="upper right")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "04a_ctrl_timeseries.png", dpi=180)
plt.close(fig1)
print("✓ 04a saved")

# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Tick-boundary discontinuity heatmap
# ═══════════════════════════════════════════════════════════════════════
# jump[i] = ctrl[i+1, 0, :] - ctrl[i, -1, :]   for i in 0..15
n_boundaries = n_ticks - 1
jumps = np.zeros((n_act, n_boundaries))
for i in range(n_boundaries):
    jumps[:, i] = np.abs(ctrl[i + 1, 0, :] - ctrl[i, -1, :])

fig2, ax2 = plt.subplots(figsize=(12, 8))
im = ax2.imshow(jumps, aspect="auto", cmap="hot", interpolation="nearest")
ax2.set_xlabel("Tick boundary (between tick i → i+1)")
ax2.set_ylabel("Actuator index")
ax2.set_xticks(range(n_boundaries))
ax2.set_xticklabels([f"{i}→{i+1}" for i in range(n_boundaries)], fontsize=7, rotation=45)
ax2.set_yticks(range(0, n_act, 3))
ax2.set_title("|Δctrl| at Tick Boundaries")
cb = fig2.colorbar(im, ax=ax2, shrink=0.8)
cb.set_label("|jump|")

# annotate max jump per boundary
for b in range(n_boundaries):
    max_act = np.argmax(jumps[:, b])
    max_val = jumps[max_act, b]
    if max_val > 0.05:
        ax2.text(b, max_act, f"{max_val:.2f}", ha="center", va="center",
                 fontsize=5, color="cyan", fontweight="bold")

fig2.tight_layout()
fig2.savefig(OUT_DIR / "04b_ctrl_discontinuity.png", dpi=180)
plt.close(fig2)
print("✓ 04b saved")

# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Control effort per tick + opt_steps overlay
# ═══════════════════════════════════════════════════════════════════════
# Mean L2 norm of ctrl across sub-steps within each tick
ctrl_l2 = np.linalg.norm(ctrl, axis=2)            # (17, 40) — L2 over 36 actuators
ctrl_effort = ctrl_l2.mean(axis=1)                 # (17,)    — mean over 40 steps

# Mean |Δctrl| (rate of change) between consecutive sub-steps
dctrl = np.diff(ctrl, axis=1)                      # (17, 39, 36)
dctrl_l2 = np.linalg.norm(dctrl, axis=2)           # (17, 39)
ctrl_rate = dctrl_l2.mean(axis=1)                   # (17,)

ticks = np.arange(n_ticks)

fig3, ax3a = plt.subplots(figsize=(12, 5))
bar_w = 0.35
b1 = ax3a.bar(ticks - bar_w / 2, ctrl_effort, bar_w, label="Mean ‖ctrl‖₂", color="steelblue", alpha=0.85)
b2 = ax3a.bar(ticks + bar_w / 2, ctrl_rate, bar_w, label="Mean ‖Δctrl‖₂ (rate)", color="salmon", alpha=0.85)
ax3a.set_xlabel("Tick")
ax3a.set_ylabel("Control magnitude / rate")
ax3a.set_xticks(ticks)
ax3a.legend(loc="upper left", fontsize=8)
ax3a.grid(True, axis="y", alpha=0.3)

# overlay opt_steps on secondary y-axis
ax3b = ax3a.twinx()
ax3b.plot(ticks, opt_steps, "k-o", ms=4, lw=1.5, label="opt_steps")
ax3b.set_ylabel("opt_steps (optimizer iterations)")
ax3b.legend(loc="upper right", fontsize=8)

ax3a.set_title("Per-Tick Control Effort vs Optimizer Steps")
fig3.tight_layout()
fig3.savefig(OUT_DIR / "04c_ctrl_effort.png", dpi=180)
plt.close(fig3)
print("✓ 04c saved")

# ── summary stats ────────────────────────────────────────────────────
print(f"\n— Summary —")
print(f"  Max boundary jump: {jumps.max():.4f}  (boundary {jumps.max(axis=0).argmax()}, actuator {jumps.max(axis=1).argmax()})")
print(f"  Mean boundary jump: {jumps.mean():.4f}")
print(f"  Ctrl effort range: {ctrl_effort.min():.3f} – {ctrl_effort.max():.3f}")
print(f"  Ctrl rate range:   {ctrl_rate.min():.4f} – {ctrl_rate.max():.4f}")
print(f"  opt_steps range:   {opt_steps.min()} – {opt_steps.max()}")
print(f"  Correlation(ctrl_effort, opt_steps): {np.corrcoef(ctrl_effort, opt_steps)[0,1]:.3f}")
print(f"  Correlation(ctrl_rate, opt_steps):   {np.corrcoef(ctrl_rate, opt_steps)[0,1]:.3f}")
print("\nDone ✓")
