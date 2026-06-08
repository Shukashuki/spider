#!/usr/bin/env python3
"""Quick MPPI-CMA benchmark: focus ticks only, reduced samples/seeds.

Runs ticks 0-3 with 1 iteration (cheap fast-forward), then ticks 4-6
with full 64 iterations. 512 samples, 3 seeds, 2 conditions × 2 optimizers.

Usage:
    cd /path/to/spider
    bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate spider && \
        PYTHONUNBUFFERED=1 python -u experiments/run_mppi_cma_quick.py'
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress the ccd_iterations spam (Python-level)
warnings.filterwarnings("ignore", message=".*ccd_iterations.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config, process_config, compute_noise_schedule, compute_steps
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.optimizers.mppi_cma import (
    make_optimize_fn_mppi_cma,
    make_optimize_once_fn_mppi_cma,
)
from spider.simulators.mjwp import (
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


# ── Only 2 conditions that matter most ──
# Gain decay = noise schedule via final_noise_scale (0.1 → exponential decay)
# No gain decay = final_noise_scale=1.0 (constant noise)
# Both use contact_guidance=False (scene.xml exists for this task)
CONDITIONS = {
    "gain_decay": {
        "label": "Gain Decay",
        "contact_guidance": False,
        "contact_rew_scale": 0.0,
        "final_noise_scale": 0.1,
    },
    "no_gain_decay": {
        "label": "No Gain Decay",
        "contact_guidance": False,
        "contact_rew_scale": 0.0,
        "final_noise_scale": 1.0,
    },
}

OPTIMIZERS = ["mppi", "mppi_cma"]
FOCUS_TICKS = [4, 5, 6]
WARMUP_ITERS = 1      # cheap iterations for ticks before focus
FULL_ITERS = 64       # full iterations for focus ticks
NUM_SAMPLES = 512


def make_config(
    task: str,
    dataset_name: str,
    seed: int,
    optimizer_type: str,
    contact_guidance: bool,
    contact_rew_scale: float,
    final_noise_scale: float = 0.1,
    max_iters: int = FULL_ITERS,
    num_samples: int = NUM_SAMPLES,
) -> Config:
    config = Config(
        robot_type="xhand",
        embodiment_type="bimanual",
        task=task,
        dataset_name=dataset_name,
        dataset_dir=os.path.join(os.path.dirname(__file__), "..", "example_datasets"),
        data_id=0,
        seed=seed,
        sim_dt=0.01,
        ctrl_dt=0.1,
        horizon=0.3,
        knot_dt=0.1,
        optimizer_type=optimizer_type,
        num_samples=num_samples,
        temperature=0.1,
        max_num_iterations=max_iters,
        improvement_threshold=-1.0,
        improvement_check_steps=999,
        contact_guidance=contact_guidance,
        contact_rew_scale=contact_rew_scale,
        cma_sigma0=0.3,
        cma_mu_ratio=0.5,
        first_ctrl_noise_scale=0.5,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=final_noise_scale,
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
    return config


def run_single_focused(config: Config, focus_ticks: list[int],
                       warmup_iters: int, full_iters: int) -> list[dict]:
    """Run sim, cheap on non-focus ticks, full optimization on focus ticks."""
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

    # Stop after last focus tick
    max_tick = max(focus_ticks) + 1
    max_sim_steps_capped = min(config.max_sim_steps, (max_tick + 1) * config.ctrl_steps)

    env = setup_env(config, ref_data)

    rollout = make_rollout_fn(
        step_env, save_state, load_state,
        get_reward, get_terminal_reward, get_terminate, get_trace,
        save_env_params, load_env_params, copy_sample_state,
    )

    # We need both optimizers ready — build for full_iters
    config.max_num_iterations = full_iters
    env_params_list = [[{}] for _ in range(full_iters)]
    config.env_params_list = env_params_list

    if config.optimizer_type == "mppi_cma":
        optimize_once_full = make_optimize_once_fn_mppi_cma(rollout)
        optimize_full = make_optimize_fn_mppi_cma(optimize_once_full)
    else:
        optimize_once_full = make_optimize_once_fn(rollout)
        optimize_full = make_optimize_fn(optimize_once_full)

    # Warmup optimizer (1 iter)
    config_warmup = Config(**{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values() if hasattr(config, f.name)})
    # Can't easily deep-copy dataclass with all fields, just swap max_num_iterations
    orig_max_iters = config.max_num_iterations

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    sim_step = 0
    tick_idx = 0

    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(config.model_path)
    mj_model.opt.timestep = float(config.sim_dt)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    # Build a warmup optimizer (always MPPI, 1 iter) so that non-focus ticks
    # produce identical trajectories regardless of optimizer_type.
    warmup_opt_once = make_optimize_once_fn(rollout)
    warmup_opt_fn = make_optimize_fn(warmup_opt_once)

    while sim_step < max_sim_steps_capped:
        is_focus = tick_idx in focus_ticks

        if is_focus:
            config.max_num_iterations = full_iters
            config.env_params_list = [[{}] for _ in range(full_iters)]

            # Use the actual optimizer only on focus ticks
            if config.optimizer_type == "mppi_cma":
                opt_once = make_optimize_once_fn_mppi_cma(rollout)
                opt_fn = make_optimize_fn_mppi_cma(opt_once)
            else:
                opt_once = make_optimize_once_fn(rollout)
                opt_fn = make_optimize_fn(opt_once)
        else:
            # Warmup: always use plain MPPI so all optimizer_types
            # arrive at the same env state when focus ticks begin.
            config.max_num_iterations = warmup_iters
            config.env_params_list = [[{}] for _ in range(warmup_iters)]
            opt_fn = warmup_opt_fn

        ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
        ctrls, infos = opt_fn(config, env, ctrls, ref_slice)

        for i in range(config.ctrl_steps):
            ctrl_step = ctrls[i]
            step_env(config, env, ctrl_step)
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
            mj_data.time += config.sim_dt

        sync_env(config, env, mj_data)
        sim_step = int(np.round(mj_data.time / config.sim_dt))

        # Receding horizon
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
            n_iters_used = full_iters
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

    COLORS = {"gain_decay": "#e74c3c", "no_gain_decay": "#3498db"}
    LINESTYLES = {"mppi": "-", "mppi_cma": "--"}

    # ── Fig 1: Convergence curves per focus tick ──
    fig, axes = plt.subplots(len(FOCUS_TICKS), 2, figsize=(14, 4 * len(FOCUS_TICKS)),
                             squeeze=False)

    for row, focus_tick in enumerate(FOCUS_TICKS):
        for col, opt in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            for cond_name, cond_meta in CONDITIONS.items():
                key = f"{cond_name}__{opt}"
                if key not in all_data:
                    continue
                d = all_data[key]
                tick_ids = d["tick_ids"]
                t_idx = np.where(tick_ids == focus_tick)[0]
                if len(t_idx) == 0:
                    continue
                t_idx = t_idx[0]
                rew = d["rew_mean"][:, t_idx, :]  # (seeds, iters)
                m = rew.mean(axis=0)
                s = rew.std(axis=0)
                iters = np.arange(n_iters)
                ax.plot(iters, m, color=COLORS[cond_name],
                        label=cond_meta["label"], linewidth=1.5)
                ax.fill_between(iters, m - s, m + s,
                                color=COLORS[cond_name], alpha=0.15)
            ax.set_title(f"Tick {focus_tick} — {opt.upper().replace('_','-')}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reward (mean)")
            if row == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Convergence: MPPI vs MPPI-CMA × Gain Decay", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_convergence.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_convergence.png")

    # ── Fig 2: Bar chart — final reward at each focus tick ──
    fig, axes = plt.subplots(1, len(FOCUS_TICKS),
                             figsize=(5 * len(FOCUS_TICKS), 5), squeeze=False)
    cond_names = list(CONDITIONS.keys())
    x = np.arange(len(cond_names))
    width = 0.35

    for col, focus_tick in enumerate(FOCUS_TICKS):
        ax = axes[0, col]
        mppi_vals, mppi_errs = [], []
        cma_vals, cma_errs = [], []
        for cond_name in cond_names:
            for opt, vals, errs in [("mppi", mppi_vals, mppi_errs),
                                     ("mppi_cma", cma_vals, cma_errs)]:
                key = f"{cond_name}__{opt}"
                if key in all_data:
                    d = all_data[key]
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

        ax.bar(x - width/2, mppi_vals, width, yerr=mppi_errs,
               label="MPPI", color="#4C72B0", capsize=3)
        ax.bar(x + width/2, cma_vals, width, yerr=cma_errs,
               label="MPPI-CMA", color="#DD8452", capsize=3)
        ax.set_title(f"Tick {focus_tick}")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITIONS[c]["label"] for c in cond_names], fontsize=9)
        ax.set_ylabel("Final Reward")
        if col == 0:
            ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("MPPI vs MPPI-CMA: Final Reward", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_bar.png")

    # ── Fig 3: Reward components at tick 5 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    focus_tick = 5
    for col, (cond_name, cond_meta) in enumerate(CONDITIONS.items()):
        for row, opt in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            key = f"{cond_name}__{opt}"
            if key not in all_data:
                continue
            d = all_data[key]
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
            if row == 0:
                ax.set_title(cond_meta["label"], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{opt.upper().replace('_','-')}\nReward")
            ax.set_xlabel("Iteration")
            if row == 0 and col == 0:
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
    parser.add_argument("--task", default="P0001_4bf4e21a-obj96945373046044")
    parser.add_argument("--dataset_name", default="hot3d")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs/mppi_cma_quick")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Starting benchmark: {len(CONDITIONS)} conds × {len(OPTIMIZERS)} opts × {args.seeds} seeds")
    log(f"Focus ticks: {FOCUS_TICKS}, samples: {NUM_SAMPLES}, full_iters: {FULL_ITERS}")
    n_iters = FULL_ITERS

    all_data = {}
    total_exps = len(CONDITIONS) * len(OPTIMIZERS)
    total_runs = total_exps * args.seeds
    run_idx = 0

    for cond_name, cond_meta in CONDITIONS.items():
        for opt in OPTIMIZERS:
            exp_key = f"{cond_name}__{opt}"
            log(f"\n{'='*60}")
            log(f"  {cond_meta['label']} | {opt.upper().replace('_','-')}")
            log(f"{'='*60}")

            seed_runs = []
            for seed in range(args.seeds):
                run_idx += 1
                log(f"  [{run_idx}/{total_runs}] seed={seed} ...")
                t0 = time.perf_counter()

                config = make_config(
                    args.task, args.dataset_name, seed, opt,
                    cond_meta["contact_guidance"],
                    cond_meta["contact_rew_scale"],
                    final_noise_scale=cond_meta.get("final_noise_scale", 0.1),
                    max_iters=FULL_ITERS,
                    num_samples=NUM_SAMPLES,
                )
                config.output_dir = os.path.join(
                    args.output_dir, exp_key, f"seed_{seed}")
                os.makedirs(config.output_dir, exist_ok=True)

                try:
                    info_list = run_single_focused(
                        config, FOCUS_TICKS, WARMUP_ITERS, FULL_ITERS)
                    seed_runs.append(info_list)
                    dt = time.perf_counter() - t0
                    ticks_str = ",".join(str(i["tick"]) for i in info_list)
                    log(f"    OK ({dt:.1f}s, ticks=[{ticks_str}])")
                except Exception as e:
                    log(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            if seed_runs:
                all_data[exp_key] = aggregate_seeds(seed_runs, n_iters)

    # Save
    for key, d in all_data.items():
        path = os.path.join(args.output_dir, f"{key}_agg.npz")
        np.savez(path, **d)
        log(f"Saved: {path}")

    # Plot
    log("\nGenerating plots...")
    plot_results(all_data, args.output_dir)
    log(f"\n✅ Done. Results in: {args.output_dir}/")
    log_f.close()


if __name__ == "__main__":
    main()
