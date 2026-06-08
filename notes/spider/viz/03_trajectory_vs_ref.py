"""
03_trajectory_vs_ref.py
Visualize SPIDER simulated trajectory vs kinematic reference.

Data:
  - trajectory_mjwp.npz: qpos (17, 40, 50) — sim trajectory, 17 ticks × 40 sub-steps × 50 DOF
  - trajectory_kinematic.npz: qpos (338, 50) at 50Hz — kinematic reference

Config:
  - sim_dt = 0.01, ref_dt = 0.02, ctrl_dt = 0.4
  - ref_steps = 2 (each ref frame → 2 sim frames)
  - 338 ref frames → 676 sim frames (sim has 680 = 17×40)

Generates:
  03a_joint_timeseries.png — Key joint qpos over time with reference overlay
  03b_tick_discontinuity.png — Heatmap of qpos jumps at tick boundaries
  03c_qpos_range.png — Heatmap of per-tick qpos range
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

# ── load data ──────────────────────────────────────────────────────────
mjwp = np.load(DATA_DIR / "trajectory_mjwp.npz")
kin = np.load(DATA_DIR / "trajectory_kinematic.npz")

qpos_sim_raw = mjwp["qpos"]  # (17, 40, 50)
time_raw = mjwp["time"]      # (17, 40)
qpos_ref_raw = kin["qpos"]   # (338, 50)
ref_freq = float(kin["frequency"])  # 50 Hz → ref_dt = 0.02s

N_TICKS, N_SUB, N_DOF = qpos_sim_raw.shape

# flatten sim trajectory to continuous timeline
qpos_sim = qpos_sim_raw.reshape(-1, N_DOF)   # (680, 50)
time_sim = time_raw.reshape(-1)               # (680,)

# build reference time axis and interpolate to sim timestamps
ref_dt = 1.0 / ref_freq  # 0.02
time_ref = np.arange(len(qpos_ref_raw)) * ref_dt  # 0, 0.02, 0.04, ...

# interpolate ref to sim timestamps (linear)
qpos_ref_interp = np.zeros_like(qpos_sim)
for d in range(N_DOF):
    qpos_ref_interp[:, d] = np.interp(time_sim, time_ref, qpos_ref_raw[:, d])

# ── select representative joints ──────────────────────────────────────
# Pick joints with high variance (skip last 7 DOFs which are mostly static/object)
# Based on std analysis: DOFs 4,5,6,16,17,25,28,29 have high variation
std_per_dof = np.std(qpos_sim, axis=0)
# Pick top 8 non-zero-std joints, preferring spread across the DOF range
active_mask = std_per_dof > 0.05
active_indices = np.where(active_mask)[0]
# Select 8 representative: spread evenly across active joints
if len(active_indices) > 8:
    selected = active_indices[np.linspace(0, len(active_indices) - 1, 8, dtype=int)]
else:
    selected = active_indices

JOINT_LABELS = {
    i: f"DOF {i}" for i in range(N_DOF)
}
# More descriptive labels based on xhand bimanual structure:
# DOFs 0-5: right wrist (pos+rot), 6-17: right fingers, 18-23: left wrist, 24-35: left fingers
# DOFs 36-43: object qpos (pos+quat × 2), 44-49: zero/unused
REGION_NAMES = {}
for i in range(6):
    REGION_NAMES[i] = f"R.Wrist-{i}"
for i in range(6, 18):
    REGION_NAMES[i] = f"R.Finger-{i-6}"
for i in range(18, 24):
    REGION_NAMES[i] = f"L.Wrist-{i-18}"
for i in range(24, 36):
    REGION_NAMES[i] = f"L.Finger-{i-24}"
for i in range(36, 50):
    REGION_NAMES[i] = f"Obj-{i-36}"

# ── Figure 1: Joint timeseries with reference ─────────────────────────
fig, axes = plt.subplots(len(selected), 1, figsize=(16, 2.5 * len(selected)),
                          sharex=True)
if len(selected) == 1:
    axes = [axes]

colors_sim = plt.cm.tab10(np.linspace(0, 1, len(selected)))

for idx, (ax, dof) in enumerate(zip(axes, selected)):
    sim_line = qpos_sim[:, dof]
    ref_line = qpos_ref_interp[:, dof]

    ax.plot(time_sim, sim_line, color=colors_sim[idx], linewidth=1.0,
            label="Sim (MJWP)", alpha=0.9)
    ax.plot(time_sim, ref_line, color="black", linewidth=1.0,
            linestyle="--", label="Ref (Kinematic)", alpha=0.6)

    # error fill
    ax.fill_between(time_sim, sim_line, ref_line,
                     alpha=0.15, color=colors_sim[idx])

    # tick boundaries
    for t in range(1, N_TICKS):
        ax.axvline(x=time_raw[t, 0], color="gray", linestyle=":", linewidth=0.5, alpha=0.6)

    label = REGION_NAMES.get(dof, f"DOF {dof}")
    ax.set_ylabel(label, fontsize=9)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Joint qpos Timeseries: Sim (MJWP) vs Reference (Kinematic)\n"
             f"Task: p36-tea | {N_TICKS} ticks × {N_SUB} sub-steps | {N_DOF} DOF",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "03a_joint_timeseries.png", dpi=150)
plt.close(fig)
print(f"✓ Saved 03a_joint_timeseries.png  ({len(selected)} joints: {selected.tolist()})")

# ── Figure 2: Tick boundary discontinuity heatmap ─────────────────────
n_boundaries = N_TICKS - 1
discontinuity = np.zeros((N_DOF, n_boundaries))

for t in range(n_boundaries):
    end_qpos = qpos_sim_raw[t, -1, :]      # last sub-step of tick t
    start_qpos = qpos_sim_raw[t + 1, 0, :] # first sub-step of tick t+1
    discontinuity[:, t] = np.abs(start_qpos - end_qpos)

fig, ax = plt.subplots(figsize=(14, 10))
# Only show active DOFs (skip zero-variance ones)
active_dof_mask = np.max(discontinuity, axis=1) > 1e-8
active_dof_indices = np.where(active_dof_mask)[0]
disc_active = discontinuity[active_dof_indices, :]

im = ax.imshow(disc_active, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.set_xlabel("Tick Boundary (t → t+1)", fontsize=11)
ax.set_ylabel("DOF Index", fontsize=11)
ax.set_xticks(range(n_boundaries))
ax.set_xticklabels([f"{t}→{t+1}" for t in range(n_boundaries)], fontsize=7, rotation=45)
ax.set_yticks(range(len(active_dof_indices)))
ax.set_yticklabels([REGION_NAMES.get(d, f"DOF {d}") for d in active_dof_indices], fontsize=7)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("|Δqpos| at tick boundary", fontsize=10)

# Annotate max discontinuity per boundary
for b in range(n_boundaries):
    max_idx = np.argmax(disc_active[:, b])
    max_val = disc_active[max_idx, b]
    if max_val > 0.01:
        ax.text(b, max_idx, f"{max_val:.3f}", ha="center", va="center",
                fontsize=5, color="white" if max_val > 0.03 else "black",
                fontweight="bold")

ax.set_title("Tick Boundary Discontinuity: |qpos_start(t+1) − qpos_end(t)|\n"
             f"Task: p36-tea | {n_boundaries} boundaries × {len(active_dof_indices)} active DOFs",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "03b_tick_discontinuity.png", dpi=150)
plt.close(fig)
print(f"✓ Saved 03b_tick_discontinuity.png  ({len(active_dof_indices)} active DOFs)")

# ── Figure 3: Per-tick qpos range heatmap ─────────────────────────────
qpos_range = np.zeros((N_DOF, N_TICKS))
for t in range(N_TICKS):
    tick_data = qpos_sim_raw[t, :, :]  # (40, 50)
    qpos_range[:, t] = np.max(tick_data, axis=0) - np.min(tick_data, axis=0)

# Filter to active DOFs
active_range_mask = np.max(qpos_range, axis=1) > 1e-6
active_range_indices = np.where(active_range_mask)[0]
range_active = qpos_range[active_range_indices, :]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(range_active, aspect="auto", cmap="viridis",
               interpolation="nearest")
ax.set_xlabel("Tick Index", fontsize=11)
ax.set_ylabel("DOF Index", fontsize=11)
ax.set_xticks(range(N_TICKS))
ax.set_xticklabels(range(N_TICKS), fontsize=8)
ax.set_yticks(range(len(active_range_indices)))
ax.set_yticklabels([REGION_NAMES.get(d, f"DOF {d}") for d in active_range_indices], fontsize=7)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("qpos range (max − min) within tick", fontsize=10)

ax.set_title("Per-Tick qpos Range (max − min across 40 sub-steps)\n"
             f"Task: p36-tea | {N_TICKS} ticks × {len(active_range_indices)} active DOFs",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "03c_qpos_range.png", dpi=150)
plt.close(fig)
print(f"✓ Saved 03c_qpos_range.png  ({len(active_range_indices)} active DOFs)")

# ── Summary stats ──────────────────────────────────────────────────────
print("\n=== Summary ===")
tracking_error = np.abs(qpos_sim - qpos_ref_interp)
print(f"Tracking error (all DOFs):  mean={tracking_error.mean():.6f}, "
      f"max={tracking_error.max():.6f}, median={np.median(tracking_error):.6f}")
active_error = tracking_error[:, active_mask]
print(f"Tracking error (active DOFs): mean={active_error.mean():.6f}, "
      f"max={active_error.max():.6f}, median={np.median(active_error):.6f}")
print(f"Discontinuity (all boundaries): mean={discontinuity.mean():.6f}, "
      f"max={discontinuity.max():.6f}")
print(f"Per-tick range (all ticks): mean={qpos_range.mean():.6f}, "
      f"max={qpos_range.max():.6f}")
