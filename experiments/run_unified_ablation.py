#!/usr/bin/env python3
"""Three-way comparison: SPIDER original vs MPPI-knot vs MPPI-CMA.

Line 1: SPIDER original MPPI (horizon-space weighted mean + annealing) — the real baseline
Line 2: MPPI in knot-point space (eta_sigma=0 + annealing) — isolate space difference
Line 3: MPPI-CMA in knot-point space (eta_sigma>0 + annealing) — CMA contribution

Focus ticks 4-6, warmup ticks use shared MPPI (1 iter) so env state
is identical when focus begins.

Usage:
    cd /path/to/spider
    PYTHONUNBUFFERED=1 python -u experiments/run_unified_ablation.py [--seeds 5]
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
from spider.optimizers.mppi_unified import (
    make_optimize_fn_unified,
    make_optimize_once_fn_unified,
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


# ── Three lines ──
# "engine": "original" uses SPIDER's native MPPI (horizon-space weighted mean)
# "engine": "unified"  uses the knot-point unified optimizer
CONDITIONS = {
    "spider_original": {
        "label": "SPIDER Original (MPPI + Annealing)",
        "engine": "original",
        "eta_sigma": None,
        "final_noise_scale": 0.1,
    },
    "mppi_knot": {
        "label": "MPPI Knot-Space (η_Σ=0 + Annealing)",
        "engine": "unified",
        "eta_sigma": 0.0,
        "final_noise_scale": 0.1,
    },
    "mppi_cma": {
        "label": "MPPI-CMA Knot-Space (η_Σ=0.3 + Annealing)",
        "engine": "unified",
        "eta_sigma": 0.3,
        "final_noise_scale": 0.1,
    },
}

FOCUS_TICKS = [4, 5, 6]
WARMUP_ITERS = 1
FULL_ITERS = 64
NUM_SAMPLES = 512
ETA_MU = 0.5


def make_config(
    task: str,
    dataset_name: str,
    seed: int,
    final_noise_scale: float = 0.1,
) -> Config:
    return Config(
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
        optimizer_type="mppi",  # doesn't matter, we use unified
        num_samples=NUM_SAMPLES,
        temperature=0.1,
        max_num_iterations=FULL_ITERS,
        improvement_threshold=-1.0,
        improvement_check_steps=999,
        contact_guidance=False,
        contact_rew_scale=0.0,
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


def run_single_focused(config: Config, engine: str, eta_sigma: float | None,
                       focus_ticks: list[int]) -> list[dict]:
    """Run sim. Warmup ticks use MPPI (1 iter). Focus ticks use specified engine."""
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

    # Warmup optimizer: always plain MPPI, 1 iter
    warmup_once = make_optimize_once_fn(rollout)
    warmup_fn = make_optimize_fn(warmup_once)

    # Focus optimizer depends on engine
    if engine == "original":
        # SPIDER original: horizon-space MPPI with annealing
        focus_once_orig = make_optimize_once_fn(rollout)
        focus_fn_orig = make_optimize_fn(focus_once_orig)
    else:
        # Unified knot-point optimizer
        focus_once_uni = make_optimize_once_fn_unified(rollout)
        focus_fn_uni = make_optimize_fn_unified(focus_once_uni)

    config.env_params_list = [[{}] for _ in range(FULL_ITERS)]

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

    orig_max_iters = config.max_num_iterations

    while sim_step < max_sim_steps_capped:
        is_focus = tick_idx in focus_ticks
        ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)

        if is_focus:
            config.max_num_iterations = FULL_ITERS
            config.env_params_list = [[{}] for _ in range(FULL_ITERS)]

            if engine == "original":
                ctrls, infos = focus_fn_orig(config, env, ctrls, ref_slice)
            else:
                ctrls, infos = focus_fn_uni(
                    config, env, ctrls, ref_slice,
                    eta_sigma=eta_sigma,
                    eta_mu=ETA_MU,
                )
        else:
            # Warmup: plain MPPI, 1 iter — identical for all conditions
            config.max_num_iterations = WARMUP_ITERS
            config.env_params_list = [[{}] for _ in range(WARMUP_ITERS)]
            ctrls, infos = warmup_fn(config, env, ctrls, ref_slice)

        # Step environment
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

    # Three lines: SPIDER original (green), MPPI knot (blue), MPPI-CMA (orange)
    STYLES = {
        "spider_original": {"color": "#2ecc71", "ls": "-",  "label": "SPIDER Original (MPPI + Annealing)"},
        "mppi_knot":       {"color": "#4C72B0", "ls": "--", "label": "MPPI Knot-Space (η_Σ=0)"},
        "mppi_cma":        {"color": "#DD8452", "ls": "-",  "label": "MPPI-CMA Knot-Space (η_Σ=0.3)"},
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
            rew = d["rew_mean"][:, t_idx, :]  # (seeds, iters)
            m = rew.mean(axis=0)
            s = rew.std(axis=0)
            iters = np.arange(n_iters)
            ax.plot(iters, m, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=1.5)
            ax.fill_between(iters, m - s, m + s,
                            color=style["color"], alpha=0.1)
        ax.set_title(f"Tick {focus_tick}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward (mean)")
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Three-Way: SPIDER Original vs MPPI-Knot vs MPPI-CMA (same μ₀)", fontsize=13)
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

        ax.bar(x, vals, width, yerr=errs, color=colors, capsize=3)
        ax.set_title(f"Tick {focus_tick}")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLES[c]["label"] for c in cond_names],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Final Reward")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Final Reward: Three-Way Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_bar.png")

    # ── Fig 3: Reward components at tick 5 ──
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 4),
                             squeeze=False)
    focus_tick = 5
    for col, (cond_name, cond_meta) in enumerate(CONDITIONS.items()):
        ax = axes[0, col]
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
        ax.set_title(cond_meta["label"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        if col == 0:
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
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/unified_ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Three-way comparison: SPIDER original vs MPPI-knot vs MPPI-CMA")
    log(f"  {len(CONDITIONS)} conditions × {args.seeds} seeds")
    log(f"  Focus ticks: {FOCUS_TICKS}, samples: {NUM_SAMPLES}, iters: {FULL_ITERS}")
    log(f"  eta_mu={ETA_MU}, warmup={WARMUP_ITERS} iter (plain MPPI)")

    all_data = {}
    total_runs = len(CONDITIONS) * args.seeds
    run_idx = 0

    for cond_name, cond_meta in CONDITIONS.items():
        log(f"\n{'='*60}")
        log(f"  {cond_meta['label']} (engine={cond_meta['engine']}, eta_sigma={cond_meta['eta_sigma']})")
        log(f"{'='*60}")

        seed_runs = []
        for seed in range(args.seeds):
            run_idx += 1
            log(f"  [{run_idx}/{total_runs}] seed={seed} ...")
            t0 = time.perf_counter()

            config = make_config(
                args.task, args.dataset_name, seed,
                final_noise_scale=cond_meta["final_noise_scale"],
            )
            config.output_dir = os.path.join(args.output_dir, cond_name, f"seed_{seed}")
            os.makedirs(config.output_dir, exist_ok=True)

            try:
                info_list = run_single_focused(
                    config, cond_meta["engine"], cond_meta["eta_sigma"], FOCUS_TICKS)
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
