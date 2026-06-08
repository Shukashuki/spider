"""
SPIDER trace_ref 3D trajectory visualization — Receding Horizon edition.

Correctly separates each tick's planning horizon into:
  - Executed portion (first ctrl_steps=40 out of 160)
  - Planned-but-not-executed remainder

Outputs:
  05a_3d_all_sites.png  — Receding horizon visualization (representative sites, two views)
  05b_site_displacement.png — Per-tick displacement heatmap (executed portion only)
  05c_bimanual_paths.png — Left vs right hand actual trajectories with gap detection
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.lines import Line2D
from pathlib import Path
import yaml

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0")
DATA_PATH = DATA_DIR / "trajectory_mjwp.npz"
CONFIG_PATH = DATA_DIR / "config.yaml"
OUT_DIR = Path("notes/spider/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
trace_ref = data["trace_ref"]  # (17, 1, 1, 160, 12, 3)
qpos = data["qpos"]            # (17, 40, 50)
time_arr = data["time"]         # (17, 40)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

N_TICKS, _, _, HORIZON_STEPS, N_SITES, _ = trace_ref.shape
# Squeeze singleton dims → (17, 160, 12, 3)
tr = trace_ref[:, 0, 0, :, :, :]

# Executed steps per tick from config (ctrl_steps in horizon-step units)
EXEC_STEPS = config.get("ctrl_steps", qpos.shape[1])  # 40
print(f"trace_ref: {trace_ref.shape}")
print(f"Horizon steps: {HORIZON_STEPS}, Executed steps per tick: {EXEC_STEPS}")
print(f"Execution ratio: {EXEC_STEPS}/{HORIZON_STEPS} = {EXEC_STEPS/HORIZON_STEPS:.1%}")

# Site labels and grouping
SITE_LABELS = [
    "L-wrist", "L-thumb", "L-index", "L-middle", "L-ring", "L-pinky",
    "R-wrist", "R-thumb", "R-index", "R-middle", "R-ring", "R-pinky",
]
LEFT_SITES = list(range(0, 6))
RIGHT_SITES = list(range(6, 12))

# Representative sites for Figure 1 (avoid clutter — just 3 sites)
REPR_SITES = {
    0: "L-wrist",
    2: "L-index",
    8: "R-index",
}

# ── colormap for ticks ────────────────────────────────────────────────
tick_cmap = cm.get_cmap("viridis", N_TICKS)


def set_equal_aspect_3d(ax, points):
    """Set equal aspect ratio for 3D axes."""
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2 * 1.15
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Receding Horizon Visualization
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating Fig 1: Receding Horizon visualization …")

n_repr = len(REPR_SITES)
fig = plt.figure(figsize=(10 * 2, 9 * n_repr))
fig.suptitle(
    f"Receding Horizon — trace_ref (p36-tea)\n"
    f"{N_TICKS} ticks × {HORIZON_STEPS} planning steps, "
    f"executed: first {EXEC_STEPS} steps/tick ({EXEC_STEPS/HORIZON_STEPS:.0%})",
    fontsize=14, fontweight="bold", y=0.99,
)

view_params = [
    {"elev": 25, "azim": -55, "title": "Perspective"},
    {"elev": 90, "azim": -90, "title": "Top-Down (XY)"},
]

for si, (s_idx, s_label) in enumerate(REPR_SITES.items()):
    # Collect all points for this site to set axis limits
    site_pts = tr[:, :, s_idx, :].reshape(-1, 3)

    for vi, vp in enumerate(view_params):
        ax = fig.add_subplot(n_repr, 2, si * 2 + vi + 1, projection="3d")

        for t in range(N_TICKS):
            color = tick_cmap(t / max(N_TICKS - 1, 1))
            traj = tr[t, :, s_idx, :]  # (160, 3)

            # ── Executed portion: solid, thick ──
            exec_end = min(EXEC_STEPS, HORIZON_STEPS)
            exec_traj = traj[: exec_end + 1]
            ax.plot(
                exec_traj[:, 0], exec_traj[:, 1], exec_traj[:, 2],
                color=color, linewidth=2.0, alpha=0.9, linestyle="-",
            )

            # ── Planned-but-not-executed: dashed, thin, transparent ──
            plan_traj = traj[exec_end:]
            if len(plan_traj) > 1:
                ax.plot(
                    plan_traj[:, 0], plan_traj[:, 1], plan_traj[:, 2],
                    color=color, linewidth=0.5, alpha=0.15, linestyle="--",
                )

            # Mark start of each tick
            ax.scatter(
                *traj[0], color=color, s=28, edgecolors="k",
                linewidths=0.4, zorder=5,
            )

            # Label tick number at start (every 2 ticks, first column only)
            if vi == 0 and t % 2 == 0:
                ax.text(
                    traj[0, 0], traj[0, 1], traj[0, 2],
                    f"  t{t}", fontsize=5.5, color=color, fontweight="bold",
                    zorder=6,
                )

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title(f"Site {s_idx}: {s_label}  —  {vp['title']}", fontsize=10)
        ax.view_init(elev=vp["elev"], azim=vp["azim"])
        set_equal_aspect_3d(ax, site_pts)

# ── Legend ──
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2.0, linestyle="-",
           label="Executed"),
    Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", alpha=0.35,
           label="Planned (not executed)"),
]
fig.legend(
    handles=legend_elements, loc="lower center", ncol=2,
    fontsize=10, frameon=True, fancybox=True,
)

# Tick-index colorbar
sm = cm.ScalarMappable(cmap=tick_cmap, norm=Normalize(vmin=0, vmax=N_TICKS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.06, aspect=30)
cbar.set_label("Tick Index", fontsize=10)

plt.tight_layout(rect=[0, 0.03, 0.93, 0.97])
out1 = OUT_DIR / "05a_3d_all_sites.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out1}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Per-tick displacement heatmap (executed portion only)
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating Fig 2: Per-tick site displacement heatmap (executed only) …")

displacements = np.zeros((N_SITES, N_TICKS))
for t in range(N_TICKS):
    for s in range(N_SITES):
        start_pos = tr[t, 0, s]                    # (3,)
        end_pos = tr[t, EXEC_STEPS - 1, s]          # last executed step
        displacements[s, t] = np.linalg.norm(end_pos - start_pos)

fig, ax = plt.subplots(figsize=(13, 5.5))
im = ax.imshow(displacements, aspect="auto", cmap="magma", interpolation="nearest")
ax.set_xlabel("Tick Index", fontsize=11)
ax.set_ylabel("Site", fontsize=11)
ax.set_xticks(range(N_TICKS))
ax.set_yticks(range(N_SITES))
ax.set_yticklabels(SITE_LABELS, fontsize=8)
ax.set_title(
    f"Per-Tick Site Displacement — Executed Portion Only "
    f"(first {EXEC_STEPS}/{HORIZON_STEPS} steps)",
    fontsize=12, fontweight="bold",
)

# Annotate cell values
for s in range(N_SITES):
    for t in range(N_TICKS):
        val = displacements[s, t]
        text_color = "w" if val > displacements.max() * 0.5 else "k"
        ax.text(t, s, f"{val:.3f}", ha="center", va="center",
                fontsize=5.5, color=text_color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Displacement (m)", fontsize=10)

# Divider between left/right hand
ax.axhline(y=5.5, color="cyan", linewidth=1.5, linestyle="--", alpha=0.7)
ax.text(N_TICKS - 0.5, 2.5, "Left Hand", ha="right", va="center",
        fontsize=8, color="cyan", fontweight="bold")
ax.text(N_TICKS - 0.5, 8.5, "Right Hand", ha="right", va="center",
        fontsize=8, color="cyan", fontweight="bold")

plt.tight_layout()
out2 = OUT_DIR / "05b_site_displacement.png"
fig.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out2}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Left vs Right hand actual trajectories (executed segments)
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating Fig 3: Bimanual executed paths with gap detection …")

# Mean position across hand sites, executed steps only
left_mean = tr[:, :EXEC_STEPS, LEFT_SITES, :].mean(axis=2)   # (17, 40, 3)
right_mean = tr[:, :EXEC_STEPS, RIGHT_SITES, :].mean(axis=2)  # (17, 40, 3)

fig = plt.figure(figsize=(18, 8))
fig.suptitle(
    "Bimanual Executed Trajectories — Left vs Right Hand (mean of 6 sites)\n"
    f"{N_TICKS} ticks × {EXEC_STEPS} executed steps each, "
    "inter-tick gaps shown in red dashed",
    fontsize=12, fontweight="bold",
)

hand_configs = [
    {"data": left_mean,  "name": "Left Hand",  "cmap_name": "winter", "subplot": 1},
    {"data": right_mean, "name": "Right Hand", "cmap_name": "autumn", "subplot": 2},
]

for hc in hand_configs:
    ax = fig.add_subplot(1, 2, hc["subplot"], projection="3d")
    hand_cmap = cm.get_cmap(hc["cmap_name"], N_TICKS + 2)
    all_pts = hc["data"].reshape(-1, 3)

    gap_distances = []

    for t in range(N_TICKS):
        color = hand_cmap((t + 1) / (N_TICKS + 1))
        segment = hc["data"][t]  # (40, 3)

        # Plot executed segment as solid line
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, linewidth=1.8, alpha=0.85,
        )

        # Tick boundary markers
        ax.scatter(
            *segment[0], color=color, s=35, marker="o",
            edgecolors="k", linewidths=0.5, zorder=5,
        )
        ax.scatter(
            *segment[-1], color=color, s=18, marker="s",
            edgecolors="k", linewidths=0.3, zorder=5,
        )

        # Label tick index
        if t % 3 == 0:
            ax.text(
                segment[0, 0], segment[0, 1], segment[0, 2],
                f"  t{t}", fontsize=6, color="k", fontweight="bold",
            )

        # Inter-tick gap detection
        if t < N_TICKS - 1:
            end_pos = segment[-1]
            next_start = hc["data"][t + 1, 0]
            gap_dist = np.linalg.norm(next_start - end_pos)
            gap_distances.append(gap_dist)

            if gap_dist > 0.001:
                # Visible gap → red dashed
                ax.plot(
                    [end_pos[0], next_start[0]],
                    [end_pos[1], next_start[1]],
                    [end_pos[2], next_start[2]],
                    color="red", linewidth=1.5, linestyle="--", alpha=0.7,
                )
            else:
                # Essentially continuous → thin gray
                ax.plot(
                    [end_pos[0], next_start[0]],
                    [end_pos[1], next_start[1]],
                    [end_pos[2], next_start[2]],
                    color="gray", linewidth=0.4, linestyle="-", alpha=0.3,
                )

    # Global start / end markers
    ax.scatter(
        *hc["data"][0, 0], color="lime", s=100, marker="^",
        edgecolors="k", linewidths=1, zorder=10, label="Start (t0)",
    )
    ax.scatter(
        *hc["data"][-1, -1], color="red", s=120, marker="*",
        edgecolors="k", linewidths=1, zorder=10, label="End (t16)",
    )

    # Print gap stats
    if gap_distances:
        gd = np.array(gap_distances)
        print(f"  {hc['name']} inter-tick gaps: "
              f"mean={gd.mean():.5f}  max={gd.max():.5f}  min={gd.min():.5f}")

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_zlabel("Z (m)", fontsize=9)
    ax.set_title(hc["name"], fontsize=11)
    ax.view_init(elev=25, azim=-50)
    ax.legend(fontsize=7, loc="upper left")
    set_equal_aspect_3d(ax, all_pts)

# Tick-index colorbar
sm = cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=N_TICKS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.05)
cbar.set_label("Tick Index (color gradient on segments)", fontsize=9)

# Legend for gap types
gap_legend = [
    Line2D([0], [0], color="red", linewidth=1.5, linestyle="--",
           label="Gap (discontinuity)"),
    Line2D([0], [0], color="gray", linewidth=0.5, linestyle="-", alpha=0.4,
           label="Continuous transition"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=6, label="Tick start"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
           markersize=5, label="Tick end"),
]
fig.legend(handles=gap_legend, loc="lower center", ncol=4, fontsize=8, frameon=True)

plt.tight_layout(rect=[0, 0.05, 0.93, 0.95])
out3 = OUT_DIR / "05c_bimanual_paths.png"
fig.savefig(out3, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out3}")


# ═══════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ Summary ═══")
print(f"Ticks: {N_TICKS}, Horizon: {HORIZON_STEPS}, Executed/tick: {EXEC_STEPS}")
print(f"sim_dt={config.get('sim_dt')}, ctrl_dt={config.get('ctrl_dt')}, "
      f"horizon={config.get('horizon')}s")

# Per-site gap analysis
print("\nInter-tick gap analysis (site-level):")
for s in range(N_SITES):
    gaps = []
    for t in range(N_TICKS - 1):
        end = tr[t, EXEC_STEPS - 1, s]
        start_next = tr[t + 1, 0, s]
        gaps.append(np.linalg.norm(start_next - end))
    gaps = np.array(gaps)
    print(f"  Site {s:2d} ({SITE_LABELS[s]:>10s}): "
          f"mean={gaps.mean():.5f}  max={gaps.max():.5f}  min={gaps.min():.5f}")

# Total executed path length per site
print("\nTotal executed path length per site:")
for s in range(N_SITES):
    total_len = 0.0
    for t in range(N_TICKS):
        seg = tr[t, :EXEC_STEPS, s, :]
        diffs = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        total_len += diffs.sum()
    print(f"  Site {s:2d} ({SITE_LABELS[s]:>10s}): {total_len:.4f} m")

print("\nDone — all 3 figures generated.")
