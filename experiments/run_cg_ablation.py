#!/usr/bin/env python3
"""Four-way comparison: MPPI vs MPPI-CMA × Contact Guidance on/off.

All four share identical: μ₀, sampling space (horizon), rollout, weights.
Variables:
  1. Search method: MPPI (original) vs MPPI-CMA (covariance adaptation)
  2. Contact guidance: off vs on (kp/kd gain schedule + contact delta correction)

Usage:
    cd /path/to/spider
    PYTHONUNBUFFERED=1 python -u experiments/run_cg_ablation.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*ccd_iterations.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config, process_config
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.optimizers.mppi_cma_horizon import (
    make_optimize_fn_cma_horizon,
    make_optimize_once_fn_cma_horizon,
)
from spider.simulators.mjwp import (
    compute_contact_point_delta,
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    step_env,
    sync_env,
)

import warp as wp


CONDITIONS = {
    "mppi": {
        "label": "MPPI",
        "engine": "original",
        "contact_guidance": False,
    },
    "mppi_cg": {
        "label": "MPPI + Contact Guidance",
        "engine": "original",
        "contact_guidance": True,
    },
    "cma": {
        "label": "MPPI-CMA (η_Σ=0.3)",
        "engine": "cma_horizon",
        "contact_guidance": False,
        "eta_sigma": 0.3,
    },
    "cma_cg": {
        "label": "MPPI-CMA + Contact Guidance",
        "engine": "cma_horizon",
        "contact_guidance": True,
        "eta_sigma": 0.3,
    },
}

FOCUS_TICKS = [4, 5, 6]
WARMUP_ITERS = 1
FULL_ITERS = 32
NUM_SAMPLES = 1024


def make_config(task: str, dataset_name: str, seed: int,
                contact_guidance: bool) -> Config:
    return Config(
        robot_type="xhand",
        embodiment_type="bimanual",
        task=task,
        dataset_name=dataset_name,
        dataset_dir=os.path.join(os.path.dirname(__file__), "..", "example_datasets"),
        data_id=0,
        seed=seed,
        sim_dt=0.01,
        ctrl_dt=0.4,
        horizon=1.6,
        knot_dt=0.4,
        optimizer_type="mppi",
        num_samples=NUM_SAMPLES,
        temperature=0.1,
        max_num_iterations=FULL_ITERS,
        improvement_threshold=-1.0,
        improvement_check_steps=999,
        contact_guidance=contact_guidance,
        contact_rew_scale=0.0,  # unified reward: qpos + qvel only (fair comparison)
        cma_sigma0=0.3,
        cma_mu_ratio=0.5,
        first_ctrl_noise_scale=0.5,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=0.1,
        show_viewer=False,
        viewer="",
        save_video=False,
        save_info=True,
        save_config=False,
        save_rerun=False,
        save_metrics=False,
        use_torch_compile=False,
        init_ctrl_mode="reference",
    )


def _build_kp_kd_schedule(config, mj_model):
    """Build per-iteration kp/kd gain decay schedule for contact guidance."""
    actuator_names = config.object_actuator_names
    if not actuator_names:
        actuator_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
            for aid in config.object_actuator_ids
        ]
    base_kp = np.array([
        config.init_rot_actuator_gain if ("_rot_" in (name or ""))
        else config.init_pos_actuator_gain
        for name in actuator_names
    ], dtype=np.float32)
    base_kd = np.array([
        config.init_rot_actuator_bias if ("_rot_" in (name or ""))
        else config.init_pos_actuator_bias
        for name in actuator_names
    ], dtype=np.float32)

    kp_schedule, kd_schedule = [], []
    for i in range(config.max_num_iterations):
        decay = float(config.guidance_decay_ratio) ** i
        kp_i = base_kp * decay
        kd_i = base_kd * decay
        if i == config.max_num_iterations - 1:
            kp_i = np.zeros_like(base_kp, dtype=np.float32)
            kd_i = np.zeros_like(base_kd, dtype=np.float32)
        kp_schedule.append(kp_i)
        kd_schedule.append(kd_i)
    return kp_schedule, kd_schedule


def _build_env_params_list(config, mj_model, n_iters, cg_enabled):
    """Build env_params_list with optional contact guidance kp/kd."""
    if cg_enabled:
        kp_schedule, kd_schedule = _build_kp_kd_schedule(config, mj_model)
    env_params_list = []
    for i in range(n_iters):
        params = {}
        if cg_enabled:
            params["kp"] = kp_schedule[i]
            params["kd"] = kd_schedule[i]
        env_params_list.append([params])
    return env_params_list


def _apply_contact_delta(ctrls, ctrl_ref, sim_step, config, env,
                         contact, contact_pos, contact_offset):
    """Apply contact point delta correction to ctrls."""
    if config.contact_len <= 0:
        return ctrls

    contact_mask_step = contact[sim_step][
        contact_offset : contact_offset + config.contact_len
    ]
    contact_pos_ref_step = contact_pos[sim_step]
    site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]

    ctrls_out = ctrls
    right_delta = compute_contact_point_delta(
        contact_mask_step, contact_pos_ref_step, site_xpos,
        config.hand_contact_site_ids, config.right_contact_indices,
    )
    left_delta = compute_contact_point_delta(
        contact_mask_step, contact_pos_ref_step, site_xpos,
        config.hand_contact_site_ids, config.left_contact_indices,
    )

    ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
    if (right_delta is not None and config.right_pos_ctrl_ids
            and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]):
        ctrls_out = ctrls_out.clone()
        ctrls_out[:, config.right_pos_ctrl_ids] = (
            ref_ctrl_slice[:, config.right_pos_ctrl_ids]
            + torch.clip(right_delta, -0.01, 0.01)
        )

    if (left_delta is not None and config.left_pos_ctrl_ids
            and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]):
        if ctrls_out is ctrls:
            ctrls_out = ctrls_out.clone()
        ctrls_out[:, config.left_pos_ctrl_ids] = (
            ref_ctrl_slice[:, config.left_pos_ctrl_ids]
            + torch.clip(left_delta, -0.01, 0.01)
        )

    return ctrls_out


def run_single_focused(config: Config, engine: str, eta_sigma: float = 0.3,
                       focus_ticks: list[int] = FOCUS_TICKS) -> list[dict]:
    config = process_config(config)

    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    if (
        config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        ctrl_ref = qpos_ref[:, : config.nu]
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    max_tick = max(focus_ticks) + 1
    max_sim_steps_capped = min(config.max_sim_steps, (max_tick + 1) * config.ctrl_steps)

    env = setup_env(config, ref_data)

    rollout = make_rollout_fn(
        step_env, save_state, load_state,
        get_reward, get_terminal_reward, get_terminate, get_trace,
        save_env_params, load_env_params, copy_sample_state,
    )

    # Warmup: always plain MPPI, 1 iter, no contact guidance
    warmup_once = make_optimize_once_fn(rollout)
    warmup_fn = make_optimize_fn(warmup_once)

    # Focus optimizer
    if engine == "original":
        focus_once = make_optimize_once_fn(rollout)
        focus_fn = make_optimize_fn(focus_once)
    else:
        focus_once = make_optimize_once_fn_cma_horizon(rollout)
        focus_fn = make_optimize_fn_cma_horizon(focus_once)

    # Contact guidance setup
    mj_model = mujoco.MjModel.from_xml_path(config.model_path)
    mj_model.opt.timestep = float(config.sim_dt)
    mj_data = mujoco.MjData(mj_model)

    cg_enabled = (
        config.contact_guidance and len(config.object_actuator_ids) > 0
    )
    contact_offset = 0
    if cg_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[: config.contact_len]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    sim_step = 0
    tick_idx = 0
    orig_max_iters = config.max_num_iterations

    while sim_step < max_sim_steps_capped:
        is_focus = tick_idx in focus_ticks
        ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)

        if is_focus:
            config.max_num_iterations = FULL_ITERS
            config.env_params_list = _build_env_params_list(
                config, mj_model, FULL_ITERS, cg_enabled)

            # Apply contact delta correction if enabled
            ctrls_for_opt = ctrls
            if cg_enabled:
                ctrls_for_opt = _apply_contact_delta(
                    ctrls, ctrl_ref, sim_step, config, env,
                    contact, contact_pos, contact_offset)

            if engine == "original":
                ctrls, infos = focus_fn(config, env, ctrls_for_opt, ref_slice)
            else:
                ctrls, infos = focus_fn(
                    config, env, ctrls_for_opt, ref_slice, eta_sigma=eta_sigma)
        else:
            # Warmup: plain MPPI, 1 iter, no contact guidance
            config.max_num_iterations = WARMUP_ITERS
            config.env_params_list = [[{}] for _ in range(WARMUP_ITERS)]
            ctrls, infos = warmup_fn(config, env, ctrls, ref_slice)

        for i in range(config.ctrl_steps):
            ctrl_step = ctrls[i]
            step_env(config, env, ctrl_step)
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
            mj_data.time += config.sim_dt

        sync_env(config, env, mj_data)
        sim_step = int(np.round(mj_data.time / config.sim_dt))

        prev_ctrl = ctrls[config.ctrl_steps:]
        new_ctrl = ctrl_ref[
            sim_step + prev_ctrl.shape[0] : sim_step + prev_ctrl.shape[0] + config.ctrl_steps
        ]
        if new_ctrl.shape[0] < config.ctrl_steps:
            pad = torch.zeros(
                config.ctrl_steps - new_ctrl.shape[0], ctrl_ref.shape[1],
                device=ctrl_ref.device, dtype=ctrl_ref.dtype,
            )
            new_ctrl = torch.cat([new_ctrl, pad], dim=0)
        ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

        if is_focus:
            n_iters_used = FULL_ITERS
            info_list.append({
                "tick": tick_idx,
                "rew_mean": infos["rew_mean"][:n_iters_used],
                "rew_max": infos["rew_max"][:n_iters_used],
                "rew_min": infos["rew_min"][:n_iters_used],
                "improvement": infos["improvement"][:n_iters_used],
                "opt_steps": infos["opt_steps"],
                "qpos_rew_mean": infos.get("qpos_rew_mean", np.zeros(n_iters_used))[:n_iters_used],
                "qvel_rew_mean": infos.get("qvel_rew_mean", np.zeros(n_iters_used))[:n_iters_used],
                "contact_rew_mean": infos.get("contact_rew_mean", np.zeros(n_iters_used))[:n_iters_used],
                "qpos": mj_data.qpos.copy(),
            })

        tick_idx += 1

    config.max_num_iterations = orig_max_iters
    return info_list


def aggregate_seeds(all_runs: list[list[dict]], n_iters: int) -> dict:
    n_seeds = len(all_runs)
    n_ticks = min(len(run) for run in all_runs)

    result = {"tick_ids": np.array([all_runs[0][t]["tick"] for t in range(n_ticks)])}
    for key in ["rew_mean", "rew_max", "rew_min", "improvement",
                 "qpos_rew_mean", "qvel_rew_mean", "contact_rew_mean"]:
        arr = np.zeros((n_seeds, n_ticks, n_iters))
        for s, run in enumerate(all_runs):
            for t in range(n_ticks):
                v = run[t][key]
                arr[s, t, :len(v)] = v[:n_iters]
        result[key] = arr

    nq = all_runs[0][0]["qpos"].shape[0]
    qpos_arr = np.zeros((n_seeds, n_ticks, nq))
    for s, run in enumerate(all_runs):
        for t in range(n_ticks):
            qpos_arr[s, t] = run[t]["qpos"]
    result["qpos"] = qpos_arr
    return result


def plot_results(all_data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    n_iters = FULL_ITERS

    # 4 lines: solid=no CG, dashed=CG; blue=MPPI, orange=CMA
    STYLES = {
        "mppi":    {"color": "#4C72B0", "ls": "-",  "label": "MPPI"},
        "mppi_cg": {"color": "#4C72B0", "ls": "--", "label": "MPPI + CG"},
        "cma":     {"color": "#DD8452", "ls": "-",  "label": "MPPI-CMA"},
        "cma_cg":  {"color": "#DD8452", "ls": "--", "label": "MPPI-CMA + CG"},
    }

    # ── Fig 1: Convergence curves per focus tick ──
    fig, axes = plt.subplots(1, len(FOCUS_TICKS), figsize=(6 * len(FOCUS_TICKS), 5),
                             squeeze=False)

    for col, focus_tick in enumerate(FOCUS_TICKS):
        ax = axes[0, col]
        for cond_name, style in STYLES.items():
            if cond_name not in all_data:
                continue
            d = all_data[cond_name]
            tick_ids = d["tick_ids"]
            t_idx = np.where(tick_ids == focus_tick)[0]
            if len(t_idx) == 0:
                continue
            t_idx = t_idx[0]
            rew = d["rew_mean"][:, t_idx, :]
            m = rew.mean(axis=0)
            s = rew.std(axis=0)
            iters = np.arange(n_iters)
            ax.plot(iters, m, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=2)
            ax.fill_between(iters, m - s, m + s,
                            color=style["color"], alpha=0.1)
        ax.set_title(f"Tick {focus_tick}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward (mean)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("MPPI vs MPPI-CMA × Contact Guidance", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_convergence.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_convergence.png")

    # ── Fig 2: Bar chart ──
    fig, axes = plt.subplots(1, len(FOCUS_TICKS),
                             figsize=(5 * len(FOCUS_TICKS), 5), squeeze=False)
    cond_names = list(CONDITIONS.keys())
    x = np.arange(len(cond_names))
    width = 0.6

    for col, focus_tick in enumerate(FOCUS_TICKS):
        ax = axes[0, col]
        vals, errs, colors = [], [], []
        for cond_name in cond_names:
            style = STYLES[cond_name]
            if cond_name in all_data:
                d = all_data[cond_name]
                tick_ids = d["tick_ids"]
                t_idx = np.where(tick_ids == focus_tick)[0]
                if len(t_idx) > 0:
                    final = d["rew_mean"][:, t_idx[0], -1]
                    vals.append(final.mean())
                    errs.append(final.std())
                else:
                    vals.append(0); errs.append(0)
            else:
                vals.append(0); errs.append(0)
            colors.append(style["color"])

        ax.bar(x, vals, width, yerr=errs, color=colors, capsize=4,
               edgecolor="white", linewidth=0.5)
        ax.set_title(f"Tick {focus_tick}")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLES[c]["label"] for c in cond_names],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Final Reward")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Final Reward: MPPI vs MPPI-CMA × Contact Guidance", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_bar.png")

    # ── Fig 3: Reward components at tick 5 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    focus_tick = 5
    for idx, (cond_name, style) in enumerate(STYLES.items()):
        ax = axes[idx // 2, idx % 2]
        if cond_name not in all_data:
            continue
        d = all_data[cond_name]
        tick_ids = d["tick_ids"]
        t_idx = np.where(tick_ids == focus_tick)[0]
        if len(t_idx) == 0:
            continue
        t_idx = t_idx[0]
        iters = np.arange(n_iters)
        for comp, color, lbl in [
            ("qpos_rew_mean", "#e74c3c", "qpos"),
            ("qvel_rew_mean", "#3498db", "qvel"),
            ("contact_rew_mean", "#2ecc71", "contact"),
        ]:
            curve = d[comp][:, t_idx, :].mean(axis=0)
            ax.plot(iters, curve, color=color, label=lbl, linewidth=1.2)
        total = d["rew_mean"][:, t_idx, :].mean(axis=0)
        ax.plot(iters, total, "k-", linewidth=1.5, label="total")
        ax.set_title(style["label"], fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Reward Components at Tick {focus_tick}", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_components.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_components.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset_name", default="gigahand")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/cg_ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"MPPI vs MPPI-CMA × Contact Guidance ablation")
    log(f"  {len(CONDITIONS)} conditions × {args.seeds} seeds")
    log(f"  Focus ticks: {FOCUS_TICKS}, samples: {NUM_SAMPLES}, iters: {FULL_ITERS}")

    all_data = {}
    total_runs = len(CONDITIONS) * args.seeds
    run_idx = 0

    for cond_name, cond_meta in CONDITIONS.items():
        log(f"\n{'='*60}")
        log(f"  {cond_meta['label']} (CG={'ON' if cond_meta['contact_guidance'] else 'OFF'})")
        log(f"{'='*60}")

        seed_runs = []
        for seed in range(args.seeds):
            run_idx += 1
            log(f"  [{run_idx}/{total_runs}] seed={seed} ...")
            t0 = time.perf_counter()

            config = make_config(
                args.task, args.dataset_name, seed,
                contact_guidance=cond_meta["contact_guidance"],
            )
            config.output_dir = os.path.join(args.output_dir, cond_name, f"seed_{seed}")
            os.makedirs(config.output_dir, exist_ok=True)

            eta_sigma = cond_meta.get("eta_sigma", 0.3)

            try:
                info_list = run_single_focused(
                    config, cond_meta["engine"], eta_sigma, FOCUS_TICKS)
                seed_runs.append(info_list)
                dt = time.perf_counter() - t0
                ticks_str = ",".join(str(i["tick"]) for i in info_list)
                log(f"    OK ({dt:.1f}s, ticks=[{ticks_str}])")
            except Exception as e:
                log(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        if seed_runs:
            all_data[cond_name] = aggregate_seeds(seed_runs, FULL_ITERS)

    for key, d in all_data.items():
        path = os.path.join(args.output_dir, f"{key}_agg.npz")
        np.savez(path, **d)
        log(f"Saved: {path}")

    log("\nGenerating plots...")
    plot_results(all_data, args.output_dir)
    log(f"\n✅ Done. Results in: {args.output_dir}/")
    log_f.close()


if __name__ == "__main__":
    main()
