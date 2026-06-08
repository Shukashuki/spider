#!/usr/bin/env python3
"""Replay saved trajectories from npz and render side-by-side videos.

Renders sim (from saved qpos) and ref (from trace_ref or qpos_ref) side by side.
Produces one mp4 per optimizer (best seed by final reward).

Usage:
    cd /path/to/spider
    python experiments/render_replay.py --input_dir outputs/full_task_compare
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import yaml

os.environ["MUJOCO_GL"] = "egl"


def load_model(config_path: str, width=720, height=480):
    """Load MuJoCo model from config yaml, setting offscreen buffer size."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg["model_path"]
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    # Set offscreen framebuffer size for EGL rendering
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height
    return mj_model, cfg


def render_frame(renderer, mj_model, mj_data, label="sim"):
    """Render a single frame with label overlay."""
    mujoco.mj_forward(mj_model, mj_data)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    try:
        renderer.update_scene(mj_data, "front", options)
    except Exception:
        renderer.update_scene(mj_data, 0, options)
    img = renderer.render().copy()
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return img


def render_trajectory(npz_path: str, config_path: str, output_path: str,
                      width=720, height=480, render_every=2):
    """Render a trajectory from npz to mp4."""
    data = np.load(npz_path, allow_pickle=True)
    qpos_all = data["qpos"]  # (n_ticks, steps_per_tick, nq)
    n_ticks, steps_per_tick, nq = qpos_all.shape

    mj_model, cfg = load_model(config_path, width=width, height=height)

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Also set up ref data if we have trace_ref or can load qpos_ref
    has_ref = False
    mj_data_ref = mujoco.MjData(mj_model)

    # Try to load reference qpos from dataset
    data_path = cfg.get("data_path", "")
    if data_path and os.path.exists(data_path):
        ref_data = np.load(data_path, allow_pickle=True)
        if "qpos" in ref_data:
            qpos_ref_full = ref_data["qpos"]
            has_ref = True
    elif not data_path:
        # Try to find it from dataset_dir
        dataset_dir = cfg.get("dataset_dir", "")
        task = cfg.get("task", "")
        robot = cfg.get("robot_type", "xhand")
        embod = cfg.get("embodiment_type", "bimanual")
        data_id = cfg.get("data_id", 0)
        candidate = os.path.join(dataset_dir, "processed", cfg.get("dataset_name", ""),
                                 robot, embod, task, str(data_id), "retarget.npz")
        if os.path.exists(candidate):
            ref_data = np.load(candidate, allow_pickle=True)
            if "qpos" in ref_data:
                qpos_ref_full = ref_data["qpos"]
                has_ref = True

    sim_dt = cfg.get("sim_dt", 0.01)
    render_dt = cfg.get("render_dt", 0.02)
    render_skip = max(1, int(round(render_dt / sim_dt)))

    images = []
    frame_idx = 0
    for tick in range(n_ticks):
        for step in range(steps_per_tick):
            if frame_idx % render_skip != 0:
                frame_idx += 1
                continue

            # Set sim state
            mj_data.qpos[:nq] = qpos_all[tick, step]
            mujoco.mj_forward(mj_model, mj_data)

            sim_img = render_frame(renderer, mj_model, mj_data, label="sim")

            if has_ref:
                global_step = tick * steps_per_tick + step
                if global_step < len(qpos_ref_full):
                    mj_data_ref.qpos[:] = qpos_ref_full[global_step][:mj_data_ref.qpos.shape[0]]
                    ref_img = render_frame(renderer, mj_model, mj_data_ref, label="ref")
                    combined = np.concatenate([sim_img, ref_img], axis=1)
                else:
                    # Pad with black to keep consistent width
                    pad = np.zeros_like(sim_img)
                    combined = np.concatenate([sim_img, pad], axis=1)
            else:
                combined = sim_img

            # Add tick/step info
            info_text = f"tick {tick}/{n_ticks}  step {step}/{steps_per_tick}"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            images.append(combined)
            frame_idx += 1

    if images:
        fps = int(1 / render_dt)
        imageio.mimsave(output_path, images, fps=fps)
        print(f"  ✓ {output_path} ({len(images)} frames, {len(images)/fps:.1f}s)")
    else:
        print(f"  ✗ No frames rendered for {npz_path}")


def find_best_seed(base_dir: str, opt_name: str, n_seeds: int = 3):
    """Find the seed with best (highest) final reward."""
    best_seed = 0
    best_rew = -np.inf
    for s in range(n_seeds):
        npz = os.path.join(base_dir, opt_name, f"seed_{s}", "trajectory_mjwp.npz")
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            # Final tick, final iter reward
            n_ticks = d["rew_u0"].shape[0]
            opt_steps = int(d["opt_steps"][-1].item())
            final_rew = d["rew_u0"][-1, opt_steps - 1]
            if final_rew > best_rew:
                best_rew = final_rew
                best_seed = s
    return best_seed, best_rew


OPTIMIZERS = {
    "dial_mpc": "DIAL-MPC",
    "mppi_pure": "Pure MPPI",
    "mppi_cma_rank": "MPPI+CMA (rank, anneal)",
    "mppi_cma_rank_noanneal_s005": "MPPI+CMA (no anneal)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/full_task_compare")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    base = args.input_dir
    video_dir = os.path.join(base, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for opt_name, label in OPTIMIZERS.items():
        opt_dir = os.path.join(base, opt_name)
        if not os.path.isdir(opt_dir):
            print(f"  skip {opt_name} (not found)")
            continue

        best_seed, best_rew = find_best_seed(base, opt_name, args.seeds)
        print(f"\n[{label}] best seed={best_seed} (rew={best_rew:.4f})")

        npz_path = os.path.join(opt_dir, f"seed_{best_seed}", "trajectory_mjwp.npz")
        cfg_path = os.path.join(opt_dir, f"seed_{best_seed}", "config.yaml")
        out_path = os.path.join(video_dir, f"{opt_name}_seed{best_seed}.mp4")

        if not os.path.exists(cfg_path):
            print(f"  skip (no config.yaml)")
            continue

        render_trajectory(npz_path, cfg_path, out_path,
                          width=args.width, height=args.height)

    print(f"\n✅ Videos saved to: {video_dir}/")


if __name__ == "__main__":
    main()
