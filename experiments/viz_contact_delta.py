"""Visualize contact guidance delta as an animated video.

For each sim step, replays qpos through MuJoCo to get fingertip site_xpos,
compares with reference contact_pos, and renders:
  - Side-by-side: sim hand (left) + reference hand (right)
  - Colored arrows on active contact fingertips showing delta vectors
  - Per-finger delta norm bar chart (bottom panel)
  - Aggregate delta norm time series

Usage:
    cd spider/
    .venv/bin/python experiments/viz_contact_delta.py [--task p36-tea] [--fps 10]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# We need mujoco for forward kinematics
try:
    import mujoco
except ImportError:
    print("mujoco not available — install via: pip install mujoco")
    sys.exit(1)


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = {
    "thumb": "#e74c3c",
    "index": "#e67e22",
    "middle": "#2ecc71",
    "ring": "#3498db",
    "pinky": "#9b59b6",
}
SIDE_NAMES = ["right", "left"]


def find_contact_site_ids(mj_model, embodiment="bimanual"):
    """Replicate SPIDER's build_hand_contact_site_ids logic."""
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


def compute_deltas_for_step(
    contact_mask, contact_pos_ref, site_xpos, site_ids, contact_order
):
    """Compute per-finger delta (current - reference) and aggregate."""
    n_contacts = len(contact_order)
    per_finger = {}  # (side, finger) -> {"delta": (3,), "current": (3,), "ref": (3,), "active": bool}

    for idx in range(n_contacts):
        side, finger = contact_order[idx]
        key = (side, finger)
        sid = site_ids[idx]

        if sid is None or idx >= len(contact_mask) or contact_mask[idx] <= 0.5:
            per_finger[key] = {"delta": np.zeros(3), "active": False}
            continue

        current = site_xpos[sid]
        ref = contact_pos_ref[idx]
        delta = current - ref
        per_finger[key] = {
            "delta": delta,
            "current": current.copy(),
            "ref": ref.copy(),
            "active": True,
        }

    return per_finger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset", default="gigahand")
    parser.add_argument("--robot", default="xhand")
    parser.add_argument("--embodiment", default="bimanual")
    parser.add_argument("--data-id", default="0")
    parser.add_argument("--run-dir", default=None,
                        help="Path to trajectory_mjwp_act.npz (auto-detected if not set)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    # Paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(
        base, "example_datasets", "processed",
        args.dataset, args.robot, args.embodiment, args.task, args.data_id,
    )

    # Load kinematic reference (contact mask + positions)
    kin_path = os.path.join(dataset_dir, "trajectory_kinematic_act.npz")
    if not os.path.exists(kin_path):
        kin_path = os.path.join(dataset_dir, "trajectory_kinematic.npz")
    kin = np.load(kin_path)
    contact_mask_all = kin["contact"]       # (T_kin, N_contact)
    contact_pos_all = kin["contact_pos"]    # (T_kin, N_contact, 3)
    qpos_ref = kin["qpos"]                  # (T_kin, nq)
    print(f"Kinematic ref: contact={contact_mask_all.shape}, qpos_ref={qpos_ref.shape}")

    # Load optimized trajectory
    if args.run_dir:
        run_npz = args.run_dir
    else:
        run_npz = os.path.join(
            base, "outputs", "multi_task",
            f"{args.dataset}_{args.task}", "spider_original",
            "trajectory_mjwp_act.npz",
        )
    if not os.path.exists(run_npz):
        print(f"Run npz not found: {run_npz}")
        sys.exit(1)

    run = np.load(run_npz)
    qpos_sim = run["qpos"]  # (n_ticks, ctrl_steps, nq)
    n_ticks, ctrl_steps, nq = qpos_sim.shape
    print(f"Sim trajectory: {qpos_sim.shape} ({n_ticks} ticks × {ctrl_steps} substeps)")

    # Load MuJoCo model
    config_yaml = os.path.join(dataset_dir, "config_act.yaml")
    if not os.path.exists(config_yaml):
        config_yaml = os.path.join(dataset_dir, "config.yaml")

    import yaml
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    # model_path is the canonical key in SPIDER configs
    xml_path = cfg.get("model_path", cfg.get("xml_path", ""))
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(base, xml_path)
    # Normalize (config may have ../ segments)
    xml_path = os.path.normpath(xml_path)
    if not os.path.exists(xml_path):
        # Fallback: scene_act.xml next to dataset dir
        fallback = os.path.join(os.path.dirname(dataset_dir), "scene_act.xml")
        if os.path.exists(fallback):
            xml_path = fallback
    print(f"MuJoCo XML: {xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Find contact site IDs
    contact_order, site_ids = find_contact_site_ids(mj_model, args.embodiment)
    print(f"Contact order: {len(contact_order)} fingers")
    for idx, (side, finger) in enumerate(contact_order):
        sid = site_ids[idx]
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid) if sid is not None else "MISSING"
        print(f"  [{idx}] {side}/{finger} -> site {sid} ({name})")

    # Flatten sim trajectory to per-step
    sim_dt = cfg.get("sim_dt", 0.002)
    ctrl_dt = sim_dt * ctrl_steps  # time per tick

    # Collect per-step data
    all_deltas = []  # list of per_finger dicts
    all_norms = []   # (T, n_contacts) delta norms
    all_agg = []     # (T,) aggregate norm

    total_steps = n_ticks * ctrl_steps
    print(f"Computing contact deltas for {total_steps} steps...")

    for tick in range(n_ticks):
        for substep in range(ctrl_steps):
            sim_step = tick * ctrl_steps + substep
            # Set qpos and run forward kinematics
            qpos_step = qpos_sim[tick, substep]
            mj_data.qpos[:len(qpos_step)] = qpos_step
            mujoco.mj_kinematics(mj_model, mj_data)

            # Get contact mask/pos for this sim_step
            kin_step = min(sim_step, contact_mask_all.shape[0] - 1)
            contact_mask = contact_mask_all[kin_step]
            contact_pos_ref = contact_pos_all[kin_step]

            # Truncate to match contact_order length
            n_c = min(len(contact_order), contact_mask.shape[0])

            per_finger = compute_deltas_for_step(
                contact_mask[:n_c], contact_pos_ref[:n_c],
                mj_data.site_xpos, site_ids, contact_order,
            )
            all_deltas.append(per_finger)

            # Per-finger norms
            norms = []
            for side, finger in contact_order:
                d = per_finger[(side, finger)]
                norms.append(np.linalg.norm(d["delta"]) if d["active"] else 0.0)
            all_norms.append(norms)

            # Aggregate: mean of active deltas
            active_norms = [n for n, (s, f) in zip(norms, contact_order)
                           if per_finger[(s, f)]["active"]]
            all_agg.append(np.mean(active_norms) if active_norms else 0.0)

    all_norms = np.array(all_norms)  # (T, n_contacts)
    all_agg = np.array(all_agg)      # (T,)

    print(f"Delta stats: mean={all_agg.mean():.5f}, max={all_agg.max():.5f}")

    # --- Render video frames ---
    # Subsample: one frame per ctrl_steps (= per tick boundary + a few intra-tick)
    # For smooth video, sample every N steps
    step_stride = max(1, ctrl_steps // 4)  # ~4 frames per tick
    frame_indices = list(range(0, total_steps, step_stride))
    print(f"Rendering {len(frame_indices)} frames (stride={step_stride})...")

    output_dir = os.path.join(base, "outputs", "viz_contact_delta")
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    for fi, step_idx in enumerate(frame_indices):
        tick = step_idx // ctrl_steps
        substep = step_idx % ctrl_steps

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

        # --- Top left: per-finger delta norm bar chart ---
        ax_bar = fig.add_subplot(gs[0, 0])
        per_finger = all_deltas[step_idx]
        labels = []
        bar_vals = []
        bar_colors = []
        for side, finger in contact_order:
            d = per_finger[(side, finger)]
            short = f"{side[0].upper()}-{finger[:3]}"
            labels.append(short)
            bar_vals.append(np.linalg.norm(d["delta"]) * 1000 if d["active"] else 0)
            bar_colors.append(FINGER_COLORS[finger] if d["active"] else "#cccccc")

        bars = ax_bar.bar(range(len(labels)), bar_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=45, fontsize=7)
        ax_bar.set_ylabel("Delta norm (mm)")
        ax_bar.set_title(f"Per-finger contact delta  |  Tick {tick}, substep {substep}")
        ax_bar.set_ylim(0, max(all_norms.max() * 1000 * 1.2, 1))
        ax_bar.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="clamp ±10mm")
        ax_bar.legend(fontsize=7)

        # --- Top right: 3D scatter of fingertip positions ---
        ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
        for side, finger in contact_order:
            d = per_finger[(side, finger)]
            if not d["active"]:
                continue
            c = d["current"]
            r = d["ref"]
            color = FINGER_COLORS[finger]
            ax_3d.scatter(*c, color=color, s=40, marker="o", edgecolors="black", linewidths=0.5)
            ax_3d.scatter(*r, color=color, s=40, marker="x", alpha=0.5)
            # Arrow from ref to current
            delta = d["delta"]
            ax_3d.quiver(r[0], r[1], r[2], delta[0], delta[1], delta[2],
                        color=color, arrow_length_ratio=0.3, linewidth=1.5)

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Fingertip: ● sim  ✕ ref  → delta")

        # Auto-scale around active points
        active_pts = []
        for side, finger in contact_order:
            d = per_finger[(side, finger)]
            if d["active"]:
                active_pts.extend([d["current"], d["ref"]])
        if active_pts:
            pts = np.array(active_pts)
            center = pts.mean(axis=0)
            span = max(pts.max(axis=0) - pts.min(axis=0)) * 0.6 + 0.02
            ax_3d.set_xlim(center[0] - span, center[0] + span)
            ax_3d.set_ylim(center[1] - span, center[1] + span)
            ax_3d.set_zlim(center[2] - span, center[2] + span)

        # --- Bottom: aggregate delta norm time series ---
        ax_ts = fig.add_subplot(gs[1, :])
        time_axis = np.arange(total_steps) * sim_dt
        ax_ts.plot(time_axis, all_agg * 1000, color="#2c3e50", linewidth=0.8, alpha=0.7)
        # Highlight current position
        ax_ts.axvline(x=step_idx * sim_dt, color="red", linewidth=1.5, alpha=0.8)
        # Tick boundaries
        for t in range(n_ticks + 1):
            ax_ts.axvline(x=t * ctrl_steps * sim_dt, color="gray", linewidth=0.3, alpha=0.5)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Mean active delta (mm)")
        ax_ts.set_title("Contact guidance error over time")
        ax_ts.set_xlim(0, total_steps * sim_dt)
        ax_ts.axhline(y=10, color="red", linestyle="--", alpha=0.3, label="clamp ±10mm")

        # Per-finger traces (thin lines)
        for ci, (side, finger) in enumerate(contact_order):
            ax_ts.plot(time_axis, all_norms[:, ci] * 1000,
                      color=FINGER_COLORS[finger], linewidth=0.3, alpha=0.4)

        frame_path = os.path.join(output_dir, f"frame_{fi:04d}.png")
        fig.savefig(frame_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

        if fi % 20 == 0:
            print(f"  Frame {fi}/{len(frame_indices)}")

    # --- Compile video with imageio ---
    out_video = args.output or os.path.join(output_dir, "contact_delta.mp4")
    try:
        import imageio.v3 as iio
        frames_data = []
        for fp in frame_paths:
            frames_data.append(iio.imread(fp))
        iio.imwrite(out_video, np.stack(frames_data), fps=args.fps, codec="libx264")
        print(f"Compiled with imageio")
    except Exception as e:
        print(f"imageio failed ({e}), trying ffmpeg CLI...")
        cmd = (
            f"ffmpeg -y -framerate {args.fps} "
            f"-i {output_dir}/frame_%04d.png "
            f"-c:v libx264 -pix_fmt yuv420p -crf 23 "
            f'"{out_video}"'
        )
        os.system(cmd)

    # Cleanup frames
    for fp in frame_paths:
        os.remove(fp)

    print(f"\n✅ Video saved: {out_video}")
    print(f"   {len(frame_indices)} frames @ {args.fps} fps = {len(frame_indices)/args.fps:.1f}s")


if __name__ == "__main__":
    main()
