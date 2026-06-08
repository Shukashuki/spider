"""Replay saved trajectory (qpos from npz) and render to mp4 using run_mjwp logic.

ref on LEFT, sim on RIGHT — same layout as run_mjwp.py save_video=true.
Reference qpos is interpolated from ref_dt to sim_dt before indexing.

Usage:
    python examples/render_trajectory.py --run outputs/spoon_full/mppi
    python examples/render_trajectory.py --base outputs/spoon_full   # all subdirs
"""

from __future__ import annotations

import argparse
import os

import imageio
import mujoco
import numpy as np
import yaml


def load_config(run_dir: str) -> dict:
    for name in ["config_act.yaml", "config.yaml"]:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No config yaml in {run_dir}")


def find_npz(run_dir: str) -> str:
    for name in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No trajectory npz in {run_dir}")


def interp_qpos(qpos: np.ndarray, ref_steps: int) -> np.ndarray:
    """Linear interpolation: (T, nq) at ref_dt → (T*ref_steps, nq) at sim_dt."""
    T, nq = qpos.shape
    out_T = T * ref_steps
    result = np.empty((out_T, nq), dtype=np.float32)
    for i in range(T - 1):
        for s in range(ref_steps):
            alpha = s / ref_steps
            result[i * ref_steps + s] = (1 - alpha) * qpos[i] + alpha * qpos[i + 1]
    # last segment: repeat last frame
    for s in range(ref_steps):
        result[(T - 1) * ref_steps + s] = qpos[-1]
    return result


def render_image(renderer, mj_model, mj_data, mj_data_ref):
    """Same as spider.viewers.render_image — ref LEFT, sim RIGHT."""
    import cv2

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # ref frame (left)
    mujoco.mj_forward(mj_model, mj_data_ref)
    try:
        renderer.update_scene(mj_data_ref, "front")
    except Exception:
        renderer.update_scene(mj_data_ref, 0)
    ref_image = renderer.render().copy()
    cv2.putText(ref_image, "ref", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    # sim frame (right)
    mujoco.mj_forward(mj_model, mj_data)
    try:
        renderer.update_scene(mj_data, "front", options)
    except Exception:
        renderer.update_scene(mj_data, 0, options)
    sim_image = renderer.render().copy()
    cv2.putText(sim_image, "sim", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    return np.concatenate([ref_image, sim_image], axis=1)


def render_run(run_dir: str, out_path: str, max_ticks: int = -1):
    cfg = load_config(run_dir)
    npz_path = find_npz(run_dir)

    model_path = cfg["model_path"]
    data_path  = cfg["data_path"]
    sim_dt     = float(cfg.get("sim_dt", 0.005))
    ref_dt     = float(cfg.get("ref_dt", 0.02))
    render_dt  = float(cfg.get("render_dt", 0.02))
    render_every = max(1, round(render_dt / sim_dt))
    fps = int(round(1.0 / render_dt))

    mj_model    = mujoco.MjModel.from_xml_path(model_path)
    mj_data     = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)

    # interpolate ref qpos to sim_dt resolution (same as load_data in io.py)
    ref_steps = round(ref_dt / sim_dt)
    raw_ref_qpos = np.load(data_path)["qpos"].astype(np.float32)  # (T, nq) at 50Hz
    qpos_ref = interp_qpos(raw_ref_qpos, ref_steps)               # (T*ref_steps, nq)

    traj = np.load(npz_path)
    sim_qpos = traj["qpos"]   # (n_ticks, sim_steps_per_tick, nq)
    n_ticks, ctrl_steps, nq = sim_qpos.shape
    if max_ticks > 0:
        n_ticks = min(n_ticks, max_ticks)

    mj_model.vis.global_.offwidth  = 720
    mj_model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(mj_model, height=480, width=720)

    images = []
    sim_step = 0  # cumulative sim steps (same as run_mjwp.py)

    for tick in range(n_ticks):
        for i in range(ctrl_steps):
            if i % render_every == 0:
                mj_data.qpos[:]     = sim_qpos[tick, i, :]
                mj_data_ref.qpos[:] = qpos_ref[min(sim_step + i, len(qpos_ref) - 1)]
                images.append(render_image(renderer, mj_model, mj_data, mj_data_ref))
        sim_step += ctrl_steps

    video_path = out_path or os.path.join(run_dir, "visualization_replay.mp4")
    imageio.mimsave(video_path, images, fps=fps)
    print(f"Saved → {video_path}  ({n_ticks} ticks, {len(images)} frames, {fps}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       default=None)
    parser.add_argument("--base",      default=None)
    parser.add_argument("--out",       default=None)
    parser.add_argument("--max-ticks", type=int, default=-1)
    args = parser.parse_args()

    if args.base:
        for name in sorted(os.listdir(args.base)):
            run_dir = os.path.join(args.base, name)
            if not os.path.isdir(run_dir):
                continue
            try:
                find_npz(run_dir)
            except FileNotFoundError:
                continue
            out = os.path.join(run_dir, "visualization_replay.mp4")
            print(f"--- {name}")
            render_run(run_dir, out, max_ticks=args.max_ticks)
    elif args.run:
        render_run(args.run, args.out, max_ticks=args.max_ticks)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
