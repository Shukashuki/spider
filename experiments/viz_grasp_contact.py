"""Visualize contact guidance delta + object position for grasp quality comparison.

Shows:
  - 3D animated view: fingertip positions (sim vs ref), delta arrows, object position
  - Object z trajectory: sim vs reference
  - Per-finger delta norm bar chart
  - Aggregate contact delta time series

Supports comparing two runs (e.g., old Warp vs new Warp).

Usage:
    cd spider/
    .venv/bin/python experiments/viz_grasp_contact.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not available")

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not available")


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = {
    "thumb": "#e74c3c", "index": "#e67e22", "middle": "#2ecc71",
    "ring": "#3498db", "pinky": "#9b59b6",
}


def find_contact_site_ids(mj_model, embodiment="bimanual"):
    contact_order = []
    if embodiment in ("bimanual", "right"):
        contact_order.extend([("right", f) for f in FINGER_NAMES])
    if embodiment in ("bimanual", "left"):
        contact_order.extend([("left", f) for f in FINGER_NAMES])
    site_ids = [None] * len(contact_order)
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is None:
            continue
        name_l = name.lower()
        if "track" not in name_l or "hand" not in name_l:
            continue
        for idx, (side, finger) in enumerate(contact_order):
            if side in name_l and finger in name_l:
                if site_ids[idx] is None:
                    site_ids[idx] = sid
                break
    return contact_order, site_ids


def compute_all_deltas(qpos_sim, contact_mask_all, contact_pos_all,
                       mj_model, mj_data, contact_order, site_ids, sim_dt):
    """Compute per-step contact deltas and object positions from a trajectory."""
    n_ticks, ctrl_steps, nq = qpos_sim.shape
    total_steps = n_ticks * ctrl_steps
    n_c = min(len(contact_order), contact_mask_all.shape[1])

    all_finger_norms = np.zeros((total_steps, len(contact_order)))
    all_agg = np.zeros(total_steps)
    all_obj_z = np.zeros(total_steps)
    all_obj_xyz = np.zeros((total_steps, 3))
    # Store per-step finger data for 3D viz
    all_finger_data = []

    for tick in range(n_ticks):
        for substep in range(ctrl_steps):
            si = tick * ctrl_steps + substep
            qpos_step = qpos_sim[tick, substep]
            mj_data.qpos[:len(qpos_step)] = qpos_step
            mujoco.mj_kinematics(mj_model, mj_data)

            # Object position (qpos indices 36-38 for bimanual xhand)
            obj_start = 36
            if nq > obj_start + 2:
                all_obj_xyz[si] = qpos_step[obj_start:obj_start+3]
                all_obj_z[si] = qpos_step[obj_start + 2]

            kin_step = min(si, contact_mask_all.shape[0] - 1)
            contact_mask = contact_mask_all[kin_step]
            contact_pos_ref = contact_pos_all[kin_step]

            finger_data = {}
            active_norms = []
            for idx in range(len(contact_order)):
                side, finger = contact_order[idx]
                sid = site_ids[idx]
                key = (side, finger)

                if sid is None or idx >= n_c or contact_mask[idx] <= 0.5:
                    finger_data[key] = {"delta": np.zeros(3), "active": False}
                    continue

                current = mj_data.site_xpos[sid].copy()
                ref = contact_pos_ref[idx].copy()
                delta = current - ref
                norm = np.linalg.norm(delta)
                finger_data[key] = {
                    "delta": delta, "current": current, "ref": ref, "active": True
                }
                all_finger_norms[si, idx] = norm
                active_norms.append(norm)

            all_finger_data.append(finger_data)
            all_agg[si] = np.mean(active_norms) if active_norms else 0.0

    return all_finger_norms, all_agg, all_obj_z, all_obj_xyz, all_finger_data


def get_ref_obj_z(qpos_ref, sim_dt, ref_dt, total_steps, obj_start=36):
    """Interpolate reference object z to sim timesteps."""
    ref_z = qpos_ref[:, obj_start + 2]
    ref_times = np.arange(len(ref_z)) * ref_dt
    sim_times = np.arange(total_steps) * sim_dt
    return np.interp(sim_times, ref_times, ref_z)


def render_comparison_video(
    label_a, norms_a, agg_a, obj_z_a, obj_xyz_a, fingers_a,
    label_b, norms_b, agg_b, obj_z_b, obj_xyz_b, fingers_b,
    ref_obj_z, contact_order, sim_dt, n_ticks, ctrl_steps,
    output_dir, fps=5, dpi=100,
):
    total_steps = n_ticks * ctrl_steps
    step_stride = max(1, ctrl_steps // 2)  # ~2 frames per tick
    frame_indices = list(range(0, total_steps, step_stride))
    time_axis = np.arange(total_steps) * sim_dt

    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []

    # Precompute y-limits
    max_delta_mm = max(norms_a.max(), norms_b.max()) * 1000 * 1.15
    max_delta_mm = max(max_delta_mm, 12)  # at least show clamp line
    z_min = min(obj_z_a.min(), obj_z_b.min(), ref_obj_z.min()) - 0.01
    z_max = max(obj_z_a.max(), obj_z_b.max(), ref_obj_z.max()) + 0.02

    print(f"Rendering {len(frame_indices)} frames...")

    for fi, si in enumerate(frame_indices):
        tick = si // ctrl_steps
        substep = si % ctrl_steps

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 1.2, 1], hspace=0.35, wspace=0.3)

        # --- Row 0: 3D views side by side ---
        for col, (label, fingers, obj_xyz) in enumerate([
            (label_a, fingers_a, obj_xyz_a),
            (label_b, fingers_b, obj_xyz_b),
        ]):
            ax = fig.add_subplot(gs[0, col], projection="3d")
            fd = fingers[si]

            # Draw object as a sphere/marker
            ox, oy, oz = obj_xyz[si]
            ax.scatter(ox, oy, oz, color="#f39c12", s=200, marker="s",
                      alpha=0.7, edgecolors="black", linewidths=1, label="object", zorder=5)

            # Draw fingertips
            for side, finger in contact_order:
                d = fd[(side, finger)]
                if not d["active"]:
                    continue
                c = d["current"]
                r = d["ref"]
                color = FINGER_COLORS[finger]
                side_marker = "o" if side == "right" else "^"
                ax.scatter(*c, color=color, s=50, marker=side_marker,
                          edgecolors="black", linewidths=0.5, zorder=10)
                ax.scatter(*r, color=color, s=30, marker="x", alpha=0.4, zorder=8)
                delta = d["delta"]
                # Scale arrows for visibility
                ax.quiver(r[0], r[1], r[2], delta[0], delta[1], delta[2],
                         color=color, arrow_length_ratio=0.25, linewidth=1.5)

            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            ax.set_title(f"{label}  |  tick {tick}.{substep}", fontsize=10)
            ax.tick_params(labelsize=6)

            # Auto-scale
            all_pts = [obj_xyz[si]]
            for side, finger in contact_order:
                d = fd[(side, finger)]
                if d["active"]:
                    all_pts.extend([d["current"], d["ref"]])
            if len(all_pts) > 1:
                pts = np.array(all_pts)
                center = pts.mean(axis=0)
                span = max((pts.max(axis=0) - pts.min(axis=0)).max() * 0.6, 0.05)
                ax.set_xlim(center[0] - span, center[0] + span)
                ax.set_ylim(center[1] - span, center[1] + span)
                ax.set_zlim(center[2] - span, center[2] + span)

        # --- Row 1 left: per-finger delta bars (both runs) ---
        ax_bar = fig.add_subplot(gs[1, 0])
        labels_list = []
        for side, finger in contact_order:
            labels_list.append(f"{side[0].upper()}-{finger[:3]}")
        x = np.arange(len(labels_list))
        w = 0.35
        vals_a = norms_a[si] * 1000
        vals_b = norms_b[si] * 1000
        ax_bar.bar(x - w/2, vals_a, w, label=label_a, color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax_bar.bar(x + w/2, vals_b, w, label=label_b, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels_list, rotation=45, fontsize=7)
        ax_bar.set_ylabel("Delta (mm)", fontsize=8)
        ax_bar.set_ylim(0, max_delta_mm)
        ax_bar.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="clamp ±10mm")
        ax_bar.legend(fontsize=7, loc="upper right")
        ax_bar.set_title("Per-finger contact delta", fontsize=9)

        # --- Row 1 right: object z trajectory ---
        ax_oz = fig.add_subplot(gs[1, 1])
        ax_oz.plot(time_axis, ref_obj_z * 1000, color="black", linewidth=1.5,
                  linestyle="--", label="reference", alpha=0.7)
        ax_oz.plot(time_axis, obj_z_a * 1000, color="#3498db", linewidth=1.2, label=label_a)
        ax_oz.plot(time_axis, obj_z_b * 1000, color="#e74c3c", linewidth=1.2, label=label_b)
        ax_oz.axvline(x=si * sim_dt, color="red", linewidth=1.5, alpha=0.6)
        for t in range(n_ticks + 1):
            ax_oz.axvline(x=t * ctrl_steps * sim_dt, color="gray", linewidth=0.2, alpha=0.4)
        ax_oz.set_xlabel("Time (s)", fontsize=8)
        ax_oz.set_ylabel("Object Z (mm)", fontsize=8)
        ax_oz.set_ylim(z_min * 1000, z_max * 1000)
        ax_oz.set_xlim(0, total_steps * sim_dt)
        ax_oz.legend(fontsize=7)
        ax_oz.set_title("Object height trajectory", fontsize=9)

        # --- Row 2: aggregate delta time series ---
        ax_ts = fig.add_subplot(gs[2, :])
        ax_ts.plot(time_axis, agg_a * 1000, color="#3498db", linewidth=1, label=label_a, alpha=0.8)
        ax_ts.plot(time_axis, agg_b * 1000, color="#e74c3c", linewidth=1, label=label_b, alpha=0.8)
        ax_ts.axvline(x=si * sim_dt, color="red", linewidth=1.5, alpha=0.6)
        ax_ts.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
        for t in range(n_ticks + 1):
            ax_ts.axvline(x=t * ctrl_steps * sim_dt, color="gray", linewidth=0.2, alpha=0.4)
        ax_ts.set_xlabel("Time (s)", fontsize=8)
        ax_ts.set_ylabel("Mean contact delta (mm)", fontsize=8)
        ax_ts.set_xlim(0, total_steps * sim_dt)
        ax_ts.legend(fontsize=7)
        ax_ts.set_title("Contact guidance error over time", fontsize=9)

        frame_path = os.path.join(output_dir, f"frame_{fi:04d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

        if fi % 10 == 0:
            print(f"  Frame {fi}/{len(frame_indices)}")

    return frame_paths, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset", default="gigahand")
    parser.add_argument("--robot", default="xhand")
    parser.add_argument("--embodiment", default="bimanual")
    parser.add_argument("--data-id", default="0")
    parser.add_argument("--run-a", default=None, help="Path to npz (run A, e.g. old Warp)")
    parser.add_argument("--run-b", default=None, help="Path to npz (run B, e.g. new Warp)")
    parser.add_argument("--label-a", default="Warp 1.11-dev (broken)")
    parser.add_argument("--label-b", default="Warp 1.12.1 (fixed)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(
        base, "example_datasets", "processed",
        args.dataset, args.robot, args.embodiment, args.task, args.data_id,
    )

    # Load kinematic reference
    kin_path = os.path.join(dataset_dir, "trajectory_kinematic_act.npz")
    if not os.path.exists(kin_path):
        kin_path = os.path.join(dataset_dir, "trajectory_kinematic.npz")
    kin = np.load(kin_path)
    contact_mask_all = kin["contact"]
    contact_pos_all = kin["contact_pos"]
    qpos_ref = kin["qpos"]

    # Load config
    config_yaml = os.path.join(dataset_dir, "config_act.yaml")
    if not os.path.exists(config_yaml):
        config_yaml = os.path.join(dataset_dir, "config.yaml")
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    sim_dt = cfg.get("sim_dt", 0.01)
    ref_dt = cfg.get("ref_dt", 0.02)

    # Load MuJoCo model
    xml_path = os.path.normpath(cfg.get("model_path", cfg.get("xml_path", "")))
    print(f"MuJoCo XML: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    contact_order, site_ids = find_contact_site_ids(mj_model, args.embodiment)
    print(f"Contact sites: {len(contact_order)}")

    # Load runs
    run_a_path = args.run_a or os.path.join(
        base, "outputs", "multi_task",
        f"{args.dataset}_{args.task}", "spider_original", "trajectory_mjwp_act.npz")
    run_b_path = args.run_b or os.path.join(
        base, "outputs", "debug_kpkd_video", "trajectory_mjwp_act.npz")

    print(f"Run A: {run_a_path}")
    print(f"Run B: {run_b_path}")

    run_a = np.load(run_a_path)
    run_b = np.load(run_b_path)
    qpos_a = run_a["qpos"]
    qpos_b = run_b["qpos"]
    n_ticks, ctrl_steps, nq = qpos_a.shape
    total_steps = n_ticks * ctrl_steps
    print(f"Shape: {qpos_a.shape} ({n_ticks} ticks × {ctrl_steps} substeps)")

    # Compute deltas for both runs
    print("Computing deltas for run A...")
    norms_a, agg_a, obj_z_a, obj_xyz_a, fingers_a = compute_all_deltas(
        qpos_a, contact_mask_all, contact_pos_all,
        mj_model, mj_data, contact_order, site_ids, sim_dt)

    print("Computing deltas for run B...")
    norms_b, agg_b, obj_z_b, obj_xyz_b, fingers_b = compute_all_deltas(
        qpos_b, contact_mask_all, contact_pos_all,
        mj_model, mj_data, contact_order, site_ids, sim_dt)

    # Reference object z
    ref_obj_z = get_ref_obj_z(qpos_ref, sim_dt, ref_dt, total_steps)

    print(f"\nRun A delta: mean={agg_a.mean()*1000:.1f}mm, max={agg_a.max()*1000:.1f}mm, obj_z range={obj_z_a.max()-obj_z_a.min():.4f}")
    print(f"Run B delta: mean={agg_b.mean()*1000:.1f}mm, max={agg_b.max()*1000:.1f}mm, obj_z range={obj_z_b.max()-obj_z_b.min():.4f}")

    # Render
    output_dir = os.path.join(base, "outputs", "viz_grasp_contact")
    frame_paths, fps = render_comparison_video(
        args.label_a, norms_a, agg_a, obj_z_a, obj_xyz_a, fingers_a,
        args.label_b, norms_b, agg_b, obj_z_b, obj_xyz_b, fingers_b,
        ref_obj_z, contact_order, sim_dt, n_ticks, ctrl_steps,
        output_dir, fps=args.fps, dpi=args.dpi,
    )

    # Compile video
    out_video = args.output or os.path.join(output_dir, "grasp_contact_comparison.mp4")
    try:
        import imageio.v3 as iio
        frames_data = [iio.imread(fp) for fp in frame_paths]
        iio.imwrite(out_video, np.stack(frames_data), fps=fps, codec="libx264")
    except Exception as e:
        print(f"imageio failed ({e}), trying ffmpeg...")
        os.system(
            f'ffmpeg -y -framerate {fps} -i {output_dir}/frame_%04d.png '
            f'-c:v libx264 -pix_fmt yuv420p -crf 23 "{out_video}"'
        )

    # Cleanup
    for fp in frame_paths:
        os.remove(fp)

    print(f"\n✅ Video: {out_video}")
    print(f"   {len(frame_paths)} frames @ {fps} fps = {len(frame_paths)/fps:.1f}s")


if __name__ == "__main__":
    main()
