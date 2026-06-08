#!/usr/bin/env python3
"""Three-way comparison v3: Shared warmup → identical initial state per optimizer.

Fixes the GPU non-determinism issue from v2 by running warmup ONCE per seed,
saving env state at each focus tick, then loading the same state for all 3 optimizers.

Conditions:
  1. mppi           — vanilla MPPI
  2. mppi_cma_full  — MPPI softmax weighting + full covariance adaptation
  3. cma_es_full    — True CMA-ES: rank-based selection + full covariance

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate spider
    cd /home/roy/.openclaw/workspace/spider
    PYTHONUNBUFFERED=1 python -u experiments/run_three_way_v3.py [--seeds 5] [--iters 64]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
import mujoco
import warp as wp
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
from spider.optimizers.mppi_cma_full import (
    make_optimize_fn_mppi_cma_full,
    make_optimize_once_fn_mppi_cma_full,
)
from spider.optimizers.cma_full import (
    make_optimize_fn_cma_full,
    make_optimize_once_fn_cma_full,
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


FOCUS_TICKS = [4, 5, 6]
WARMUP_ITERS = 1


CONDITIONS = {
    "mppi": {
        "label": "MPPI (vanilla)",
        "engine": "mppi",
    },
    "mppi_cma_full": {
        "label": "MPPI + CMA-ES (full Σ)",
        "engine": "mppi_cma_full",
    },
    "cma_es_full": {
        "label": "CMA-ES (full Σ, rank-based)",
        "engine": "cma_es_full",
    },
}


def make_config(task, dataset_name, seed, full_iters, num_samples):
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
        num_samples=num_samples,
        temperature=0.1,
        max_num_iterations=full_iters,
        improvement_threshold=-1.0,
        improvement_check_steps=999,
        contact_guidance=False,
        contact_rew_scale=0.0,
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


def run_shared_warmup(config: Config, focus_ticks, full_iters):
    """Run warmup once, save snapshots at each focus tick.

    Returns list of snapshots: [{tick, env_state, mj_qpos, mj_qvel, mj_ctrl, mj_time, ctrls, sim_step}, ...]
    Also returns (config, env, ref_data, rollout, ctrl_ref) for reuse.
    """
    config = process_config(config)

    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
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

    warmup_once = make_optimize_once_fn(rollout)
    warmup_fn = make_optimize_fn(warmup_once)

    mj_model = mujoco.MjModel.from_xml_path(config.model_path)
    mj_model.opt.timestep = float(config.sim_dt)
    mj_data = mujoco.MjData(mj_model)

    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    ctrls = ctrl_ref[: config.horizon_steps]
    snapshots = []
    sim_step = 0
    tick_idx = 0

    while sim_step < max_sim_steps_capped:
        is_focus = tick_idx in focus_ticks

        if is_focus:
            # Save snapshot BEFORE optimization at this tick
            env_state = save_state(env)
            snapshots.append({
                "tick": tick_idx,
                "env_state": env_state,
                "mj_qpos": mj_data.qpos.copy(),
                "mj_qvel": mj_data.qvel.copy(),
                "mj_ctrl": mj_data.ctrl.copy(),
                "mj_time": mj_data.time,
                "ctrls": ctrls.clone(),
                "sim_step": sim_step,
            })

        # Always use MPPI warmup (1 iter) to advance simulation
        config.max_num_iterations = WARMUP_ITERS
        config.env_params_list = [[{}] for _ in range(WARMUP_ITERS)]
        ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
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

        tick_idx += 1

    return snapshots, config, env, ref_data, rollout, ctrl_ref


def run_optimizer_from_snapshot(config, env, ref_data, rollout, ctrl_ref,
                                snapshot, engine, full_iters):
    """Load a snapshot and run one optimizer for full_iters iterations.

    Returns info dict with reward curves.
    """
    tick = snapshot["tick"]
    sim_step = snapshot["sim_step"]

    # Restore env state
    load_state(env, snapshot["env_state"])

    # Restore ctrls
    ctrls = snapshot["ctrls"].clone()

    # Set up optimizer
    if engine == "mppi_cma_full":
        focus_once = make_optimize_once_fn_mppi_cma_full(rollout)
        focus_fn = make_optimize_fn_mppi_cma_full(focus_once)
    elif engine == "cma_es_full":
        focus_once = make_optimize_once_fn_cma_full(rollout)
        focus_fn = make_optimize_fn_cma_full(focus_once)
    else:
        focus_once = make_optimize_once_fn(rollout)
        focus_fn = make_optimize_fn(focus_once)

    # Run optimization
    config.max_num_iterations = full_iters
    config.env_params_list = [[{}] for _ in range(full_iters)]

    ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)

    if engine == "mppi_cma_full":
        ctrls_opt, infos = focus_fn(config, env, ctrls, ref_slice, eta_sigma=0.3)
    else:
        ctrls_opt, infos = focus_fn(config, env, ctrls, ref_slice)

    return {
        "tick": tick,
        "rew_mean": infos["rew_mean"][:full_iters],
        "rew_max": infos["rew_max"][:full_iters],
        "rew_min": infos["rew_min"][:full_iters],
        "improvement": infos["improvement"][:full_iters],
        "opt_steps": infos["opt_steps"],
        "qpos_rew_mean": infos.get("qpos_rew_mean", np.zeros(full_iters))[:full_iters],
        "qvel_rew_mean": infos.get("qvel_rew_mean", np.zeros(full_iters))[:full_iters],
        "contact_rew_mean": infos.get("contact_rew_mean", np.zeros(full_iters))[:full_iters],
        "mj_qpos": snapshot["mj_qpos"],
    }


def aggregate_seeds(all_runs, n_iters):
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

    nq = all_runs[0][0]["mj_qpos"].shape[0]
    qpos_arr = np.zeros((n_seeds, n_ticks, nq))
    for s, run in enumerate(all_runs):
        for t in range(n_ticks):
            qpos_arr[s, t] = run[t]["mj_qpos"]
    result["qpos"] = qpos_arr
    return result


def plot_results(all_data, output_dir, full_iters):
    os.makedirs(output_dir, exist_ok=True)

    STYLES = {
        "mppi":            {"color": "#DD8452", "ls": "-",  "label": "MPPI (vanilla)"},
        "mppi_cma_full":   {"color": "#55A868", "ls": "-",  "label": "MPPI + CMA-ES (full Σ)"},
        "cma_es_full":     {"color": "#4C72B0", "ls": "-",  "label": "CMA-ES (full Σ, rank-based)"},
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
            iters = np.arange(full_iters)
            ax.plot(iters, m, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=2)
            ax.fill_between(iters, m - s, m + s, color=style["color"], alpha=0.15)
        ax.set_title(f"Tick {focus_tick}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward (mean)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Three-Way: MPPI vs MPPI+CMA-ES vs CMA-ES (shared warmup)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_convergence.png")

    # ── Fig 2: Bar chart (final reward) ──
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
                           rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Final Reward")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Final Reward: Three-Way Comparison (shared warmup)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_bar.png")

    # ── Fig 3: rew_max convergence ──
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
            rew = d["rew_max"][:, t_idx, :]
            m = rew.mean(axis=0)
            s = rew.std(axis=0)
            iters = np.arange(full_iters)
            ax.plot(iters, m, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=2)
            ax.fill_between(iters, m - s, m + s, color=style["color"], alpha=0.15)
        ax.set_title(f"Tick {focus_tick}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward (max sample)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Best Sample Reward: Three-Way (shared warmup)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_rew_max.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_rew_max.png")

    # ── Fig 4: Reward components at tick 5 ──
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 5),
                             squeeze=False)
    focus_tick = 5
    for idx, (cond_name, style) in enumerate(STYLES.items()):
        ax = axes[0, idx]
        if cond_name not in all_data:
            continue
        d = all_data[cond_name]
        tick_ids = d["tick_ids"]
        t_idx = np.where(tick_ids == focus_tick)[0]
        if len(t_idx) == 0:
            continue
        t_idx = t_idx[0]
        iters = np.arange(full_iters)
        for comp, color, lbl in [
            ("qpos_rew_mean", "#e74c3c", "qpos"),
            ("qvel_rew_mean", "#3498db", "qvel"),
            ("contact_rew_mean", "#2ecc71", "contact"),
        ]:
            curve = d[comp][:, t_idx, :].mean(axis=0)
            ax.plot(iters, curve, color=color, label=lbl, linewidth=1.2)
        total = d["rew_mean"][:, t_idx, :].mean(axis=0)
        ax.plot(iters, total, "k-", linewidth=1.5, label="total")
        ax.set_title(style["label"], fontweight="bold", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Reward Components at Tick {focus_tick} (shared warmup)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_components.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig4_components.png")

    # ── Summary table ──
    print("\n" + "=" * 90)
    print(f"{'Condition':<40} {'Tick':>5} {'Final Rew (mean±std)':>25} {'Best Rew':>12}")
    print("-" * 90)
    for cond_name in cond_names:
        if cond_name not in all_data:
            continue
        d = all_data[cond_name]
        tick_ids = d["tick_ids"]
        for focus_tick in FOCUS_TICKS:
            t_idx = np.where(tick_ids == focus_tick)[0]
            if len(t_idx) == 0:
                continue
            t_idx = t_idx[0]
            final_rew = d["rew_mean"][:, t_idx, -1]
            best_rew = d["rew_max"][:, t_idx, -1]
            label = STYLES[cond_name]["label"]
            print(f"{label:<40} {focus_tick:>5} {final_rew.mean():>10.4f} ± {final_rew.std():<10.4f} {best_rew.mean():>10.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset_name", default="gigahand")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--output_dir", default="outputs/three_way_v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Three-Way v3 (shared warmup): MPPI vs MPPI+CMA-ES(full) vs CMA-ES(full)")
    log(f"  {len(CONDITIONS)} conditions × {args.seeds} seeds × {args.iters} iters")
    log(f"  Focus ticks: {FOCUS_TICKS}, samples: {args.num_samples}")

    all_cond_seed_runs = {cond: [] for cond in CONDITIONS}
    total_runs = args.seeds * (1 + len(CONDITIONS) * len(FOCUS_TICKS))
    run_idx = 0

    for seed in range(args.seeds):
        log(f"\n{'='*60}")
        log(f"  Seed {seed}: Running shared warmup...")
        log(f"{'='*60}")
        t0 = time.perf_counter()

        config = make_config(
            args.task, args.dataset_name, seed,
            full_iters=args.iters,
            num_samples=args.num_samples,
        )
        config.output_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(config.output_dir, exist_ok=True)

        try:
            snapshots, config, env, ref_data, rollout, ctrl_ref = run_shared_warmup(
                config, FOCUS_TICKS, args.iters)
            dt = time.perf_counter() - t0
            log(f"  Warmup done ({dt:.1f}s), {len(snapshots)} snapshots saved")
        except Exception as e:
            log(f"  WARMUP FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Now run each optimizer from each snapshot
        for cond_name, cond_meta in CONDITIONS.items():
            tick_results = []
            for snap in snapshots:
                run_idx += 1
                tick = snap["tick"]
                log(f"  [{cond_name}] tick={tick}, seed={seed} ...")
                t0 = time.perf_counter()

                try:
                    result = run_optimizer_from_snapshot(
                        config, env, ref_data, rollout, ctrl_ref,
                        snap, cond_meta["engine"], args.iters)
                    tick_results.append(result)
                    dt = time.perf_counter() - t0
                    log(f"    OK ({dt:.1f}s)")
                except Exception as e:
                    log(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            if tick_results:
                all_cond_seed_runs[cond_name].append(tick_results)

    # Aggregate and save
    all_data = {}
    for cond_name, seed_runs in all_cond_seed_runs.items():
        if seed_runs:
            all_data[cond_name] = aggregate_seeds(seed_runs, args.iters)
            path = os.path.join(args.output_dir, f"{cond_name}_agg.npz")
            np.savez(path, **all_data[cond_name])
            log(f"Saved: {path}")

    # Verify initial rewards match
    log("\n--- Initial Reward Verification ---")
    for tick_i, focus_tick in enumerate(FOCUS_TICKS):
        log(f"Tick {focus_tick}:")
        for cond_name in CONDITIONS:
            if cond_name in all_data:
                d = all_data[cond_name]
                init_rew = d["rew_mean"][:, tick_i, 0]
                log(f"  {cond_name:20s} iter0: mean={init_rew.mean():.4f} std={init_rew.std():.4f} vals={init_rew}")

    log("\nGenerating plots...")
    plot_results(all_data, args.output_dir, args.iters)
    log(f"\n✅ Done. Results in: {args.output_dir}/")
    log_f.close()


if __name__ == "__main__":
    main()
