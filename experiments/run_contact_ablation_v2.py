#!/usr/bin/env python3
"""Contact guidance ablation v2: 3 conditions × 2 optimizers.

NOTE: run with `python -u` or PYTHONUNBUFFERED=1 to see progress in log files.

Experiment matrix:
┌──────────┬─────────────────┬──────────────────────┬───────────────────────┐
│          │ No Contact Info │ 原版 contact guidance │ Contact Delta in Cost │
├──────────┼─────────────────┼──────────────────────┼───────────────────────┤
│ MPPI     │ pure dynamics   │ baseline (gain decay) │ cost augmentation     │
│ MPPI-CMA │ pure dynamics   │ gain decay            │ full combo            │
└──────────┴─────────────────┴──────────────────────┴───────────────────────┘

- "No Contact Info": contact_guidance=False, contact_rew_scale=0.0
  → pure MPPI/MPPI-CMA on raw dynamics, no gain decay, no contact cost
- "原版 contact guidance": contact_guidance=True, contact_rew_scale=0.0
  → gain decay + feedforward delta (SPIDER original)
- "Contact Delta in Cost": contact_guidance=True, contact_rew_scale=1.0
  → gain decay + feedforward delta + contact cost in MPPI reward

Usage:
    cd /path/to/spider
    .venv/bin/python experiments/run_contact_ablation_v2.py [--task p36-tea] [--seeds 10]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config, process_config
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


# ── 3 conditions ──
CONDITIONS = {
    "no_contact": {
        "label": "No Contact Info",
        "contact_guidance": False,
        "contact_rew_scale": 0.0,
    },
    "original": {
        "label": "Original Contact Guidance",
        "contact_guidance": True,
        "contact_rew_scale": 0.0,
    },
    "cost_augmented": {
        "label": "Contact Delta in Cost",
        "contact_guidance": True,
        "contact_rew_scale": 1.0,
    },
}

OPTIMIZERS = ["mppi", "mppi_cma"]
FOCUS_TICKS = [4, 5, 6]


def make_config(
    task: str,
    dataset_name: str,
    seed: int,
    optimizer_type: str,
    contact_guidance: bool,
    contact_rew_scale: float,
) -> Config:
    """Build a Config for this experiment."""
    return Config(
        robot_type="xhand",
        embodiment_type="bimanual",
        task=task,
        dataset_name=dataset_name,
        dataset_dir=os.path.join(os.path.dirname(__file__), "..", "example_datasets"),
        data_id=0,
        seed=seed,
        # Short horizon
        sim_dt=0.01,
        ctrl_dt=0.1,
        horizon=0.3,
        knot_dt=0.1,
        # Optimizer
        optimizer_type=optimizer_type,
        num_samples=2048,
        temperature=0.1,
        max_num_iterations=64,
        improvement_threshold=-1.0,
        improvement_check_steps=999,
        # Contact
        contact_guidance=contact_guidance,
        contact_rew_scale=contact_rew_scale,
        # CMA params
        cma_sigma0=0.3,
        cma_mu_ratio=0.5,
        # Noise
        first_ctrl_noise_scale=0.5,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=0.1,
        # No viewer
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


def run_single(config: Config) -> list[dict]:
    """Run optimization loop, return per-tick info list."""
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

    env = setup_env(config, ref_data)

    # Build env_params_list
    env_params_list = []
    for i in range(config.max_num_iterations):
        env_params_list.append([{}])
    config.env_params_list = env_params_list

    # Build optimizer
    rollout = make_rollout_fn(
        step_env, save_state, load_state,
        get_reward, get_terminal_reward, get_terminate, get_trace,
        save_env_params, load_env_params, copy_sample_state,
    )

    if config.optimizer_type == "mppi_cma":
        optimize_once = make_optimize_once_fn_mppi_cma(rollout)
        optimize = make_optimize_fn_mppi_cma(optimize_once)
    else:
        optimize_once = make_optimize_once_fn(rollout)
        optimize = make_optimize_fn(optimize_once)

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    sim_step = 0

    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(config.model_path)
    mj_model.opt.timestep = float(config.sim_dt)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    while sim_step < config.max_sim_steps:
        ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
        ctrls, infos = optimize(config, env, ctrls, ref_slice)

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

        info_list.append({
            "tick": sim_step // config.ctrl_steps,
            "rew_mean": infos["rew_mean"],
            "rew_max": infos["rew_max"],
            "rew_min": infos["rew_min"],
            "improvement": infos["improvement"],
            "opt_steps": infos["opt_steps"],
            "qpos_rew_mean": infos.get("qpos_rew_mean", np.zeros_like(infos["rew_mean"])),
            "qvel_rew_mean": infos.get("qvel_rew_mean", np.zeros_like(infos["rew_mean"])),
            "contact_rew_mean": infos.get("contact_rew_mean", np.zeros_like(infos["rew_mean"])),
            "qpos": mj_data.qpos.copy(),
        })

    return info_list


def aggregate_seeds(all_runs: list[list[dict]], n_iters: int) -> dict:
    """Aggregate info_lists across seeds."""
    n_seeds = len(all_runs)
    n_ticks = min(len(run) for run in all_runs)

    result = {}
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


# ── Plotting ──

COND_COLORS = {
    "no_contact": "#95a5a6",       # grey — no info baseline
    "original": "#3498db",          # blue — SPIDER original
    "cost_augmented": "#e74c3c",    # red — cost augmentation
}
OPT_STYLES = {"mppi": "-", "mppi_cma": "--"}
OPT_MARKERS = {"mppi": "o", "mppi_cma": "s"}


def plot_results(all_data: dict, output_dir: str, ref_qpos: np.ndarray,
                 ctrl_steps: int, nq: int, nq_obj: int):
    """Generate comparison plots for the 3×2 ablation."""
    os.makedirs(output_dir, exist_ok=True)
    n_iters = 64

    # ── Fig 1: Per-tick final reward overview ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.set_title("Final Iteration Reward per Tick (mean ± std across seeds)")
    for cond_name, cond_meta in CONDITIONS.items():
        for opt in OPTIMIZERS:
            key = f"{cond_name}__{opt}"
            if key not in all_data:
                continue
            d = all_data[key]
            final_rew = d["rew_mean"][:, :, -1]
            mean_rew = final_rew.mean(axis=0)
            std_rew = final_rew.std(axis=0)
            ticks = np.arange(len(mean_rew))
            label = f"{cond_meta['label']} [{opt.upper().replace('_','-')}]"
            ax.plot(ticks, mean_rew, OPT_STYLES[opt], color=COND_COLORS[cond_name],
                    label=label, linewidth=1.5, marker=OPT_MARKERS[opt], markersize=3)
            ax.fill_between(ticks, mean_rew - std_rew, mean_rew + std_rew,
                            color=COND_COLORS[cond_name], alpha=0.08)
    ax.axvspan(3.5, 6.5, alpha=0.06, color="red", label="Focus ticks 4-6")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Object tracking error
    ax = axes[1]
    ax.set_title("Object Position Tracking Error per Tick")
    for cond_name, cond_meta in CONDITIONS.items():
        for opt in OPTIMIZERS:
            key = f"{cond_name}__{opt}"
            if key not in all_data:
                continue
            d = all_data[key]
            qpos = d["qpos"]
            n_ticks = qpos.shape[1]
            errs = np.zeros((qpos.shape[0], n_ticks))
            for s in range(qpos.shape[0]):
                for t in range(n_ticks):
                    ref_step = min(t * ctrl_steps + ctrl_steps - 1, ref_qpos.shape[0] - 1)
                    sim_obj = qpos[s, t, nq - nq_obj:nq - nq_obj + 3]
                    ref_obj = ref_qpos[ref_step, nq - nq_obj:nq - nq_obj + 3]
                    errs[s, t] = np.linalg.norm(sim_obj - ref_obj)
            mean_err = errs.mean(axis=0)
            std_err = errs.std(axis=0)
            label = f"{cond_meta['label']} [{opt.upper().replace('_','-')}]"
            ax.plot(np.arange(n_ticks), mean_err, OPT_STYLES[opt],
                    color=COND_COLORS[cond_name], label=label, linewidth=1.5,
                    marker=OPT_MARKERS[opt], markersize=3)
            ax.fill_between(np.arange(n_ticks), mean_err - std_err, mean_err + std_err,
                            color=COND_COLORS[cond_name], alpha=0.08)
    ax.axvspan(3.5, 6.5, alpha=0.06, color="red")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Position Error (m)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_per_tick_overview.png"), dpi=150)
    plt.close()
    print(f"  Saved fig1_per_tick_overview.png")

    # ── Fig 2: Iteration convergence at focus ticks ──
    fig, axes = plt.subplots(len(FOCUS_TICKS), 2, figsize=(16, 4 * len(FOCUS_TICKS)))
    if len(FOCUS_TICKS) == 1:
        axes = axes[np.newaxis, :]

    for row, tick in enumerate(FOCUS_TICKS):
        for col, opt in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            for cond_name, cond_meta in CONDITIONS.items():
                key = f"{cond_name}__{opt}"
                if key not in all_data:
                    continue
                d = all_data[key]
                if tick >= d["rew_mean"].shape[1]:
                    continue
                rew_curve = d["rew_mean"][:, tick, :]
                mean_c = rew_curve.mean(axis=0)
                std_c = rew_curve.std(axis=0)
                iters = np.arange(len(mean_c))
                ax.plot(iters, mean_c, color=COND_COLORS[cond_name],
                        label=cond_meta["label"], linewidth=1.5)
                ax.fill_between(iters, mean_c - std_c, mean_c + std_c,
                                color=COND_COLORS[cond_name], alpha=0.12)
            ax.set_title(f"Tick {tick} — {opt.upper().replace('_','-')}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Reward (mean)")
            ax.set_xlabel("Iteration")
            if row == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Iteration Convergence at Focus Ticks (mean ± std across seeds)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_convergence_focus_ticks.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_convergence_focus_ticks.png")

    # ── Fig 3: 3×2 bar chart — MPPI vs MPPI-CMA per condition at focus ticks ──
    fig, axes = plt.subplots(1, len(FOCUS_TICKS), figsize=(5 * len(FOCUS_TICKS), 6))
    if len(FOCUS_TICKS) == 1:
        axes = [axes]

    cond_names = list(CONDITIONS.keys())
    x = np.arange(len(cond_names))
    width = 0.35

    for col, tick in enumerate(FOCUS_TICKS):
        ax = axes[col]
        mppi_vals, mppi_errs = [], []
        cma_vals, cma_errs = [], []
        for cond_name in cond_names:
            for opt, vals, errs in [("mppi", mppi_vals, mppi_errs),
                                     ("mppi_cma", cma_vals, cma_errs)]:
                key = f"{cond_name}__{opt}"
                if key in all_data and tick < all_data[key]["rew_mean"].shape[1]:
                    final = all_data[key]["rew_mean"][:, tick, -1]
                    vals.append(final.mean())
                    errs.append(final.std())
                else:
                    vals.append(0)
                    errs.append(0)

        ax.bar(x - width/2, mppi_vals, width, yerr=mppi_errs,
               label="MPPI", color="#4C72B0", capsize=3)
        ax.bar(x + width/2, cma_vals, width, yerr=cma_errs,
               label="MPPI-CMA", color="#DD8452", capsize=3)
        ax.set_title(f"Tick {tick}")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITIONS[c]["label"] for c in cond_names],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Final Reward")
        if col == 0:
            ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("MPPI vs MPPI-CMA: Final Reward at Focus Ticks", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_bar_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_bar_comparison.png")

    # ── Fig 4: Reward component breakdown at tick 5 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    tick = 5
    for col, (cond_name, cond_meta) in enumerate(CONDITIONS.items()):
        for row, opt in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            key = f"{cond_name}__{opt}"
            if key not in all_data:
                continue
            d = all_data[key]
            if tick >= d["rew_mean"].shape[1]:
                continue
            iters = np.arange(n_iters)
            for comp, color, lbl in [
                ("qpos_rew_mean", "#e74c3c", "qpos"),
                ("qvel_rew_mean", "#3498db", "qvel"),
                ("contact_rew_mean", "#2ecc71", "contact"),
            ]:
                curve = d[comp][:, tick, :].mean(axis=0)
                ax.plot(iters, curve, color=color, label=lbl, linewidth=1.2)
            total = d["rew_mean"][:, tick, :].mean(axis=0)
            ax.plot(iters, total, color="black", linewidth=1.5, label="total")
            if row == 0:
                ax.set_title(cond_meta["label"], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{opt.upper().replace('_','-')}\nReward")
            ax.set_xlabel("Iteration")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    plt.suptitle(f"Reward Component Breakdown at Tick {tick}", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_component_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig4_component_breakdown.png")

    # ── Fig 5: Optimizer improvement — MPPI-CMA vs MPPI (delta) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, (cond_name, cond_meta) in enumerate(CONDITIONS.items()):
        ax = axes[col]
        mppi_key = f"{cond_name}__mppi"
        cma_key = f"{cond_name}__mppi_cma"
        if mppi_key not in all_data or cma_key not in all_data:
            ax.set_title(f"{cond_meta['label']}\n(missing data)")
            continue
        d_mppi = all_data[mppi_key]
        d_cma = all_data[cma_key]
        n_ticks = min(d_mppi["rew_mean"].shape[1], d_cma["rew_mean"].shape[1])
        # Final reward delta: CMA - MPPI (positive = CMA better)
        mppi_final = d_mppi["rew_mean"][:, :n_ticks, -1]  # (seeds, ticks)
        cma_final = d_cma["rew_mean"][:, :n_ticks, -1]
        delta = cma_final - mppi_final  # (seeds, ticks)
        mean_delta = delta.mean(axis=0)
        std_delta = delta.std(axis=0)
        ticks = np.arange(n_ticks)
        ax.bar(ticks, mean_delta, yerr=std_delta, color=COND_COLORS[cond_name],
               capsize=2, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvspan(3.5, 6.5, alpha=0.06, color="red")
        ax.set_title(cond_meta["label"], fontsize=11)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Reward Δ (CMA − MPPI)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("MPPI-CMA Improvement over MPPI per Tick", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_cma_improvement.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig5_cma_improvement.png")


def main():
    parser = argparse.ArgumentParser(description="Contact guidance ablation v2: 3×2 matrix")
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset_name", default="gigahand")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/contact_ablation_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get reference qpos for tracking error
    ref_config = make_config(args.task, args.dataset_name, 0, "mppi", False, 0.0)
    ref_config = process_config(ref_config)
    ref_data = load_data(ref_config, ref_config.data_path)
    ref_qpos = ref_data[0].detach().cpu().numpy()
    ctrl_steps = ref_config.ctrl_steps
    nq = ref_config.nq
    nq_obj = ref_config.nq_obj
    n_iters = 64

    all_data = {}
    total_runs = len(CONDITIONS) * len(OPTIMIZERS) * args.seeds
    run_idx = 0

    for cond_name, cond_meta in CONDITIONS.items():
        for opt in OPTIMIZERS:
            exp_key = f"{cond_name}__{opt}"
            print(f"\n{'='*60}", flush=True)
            print(f"Condition: {cond_meta['label']} | Optimizer: {opt.upper().replace('_','-')}", flush=True)
            print(f"{'='*60}", flush=True)

            seed_runs = []
            for seed in range(args.seeds):
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] seed={seed} ...", end=" ", flush=True)
                t0 = time.perf_counter()

                config = make_config(
                    args.task, args.dataset_name, seed, opt,
                    cond_meta["contact_guidance"],
                    cond_meta["contact_rew_scale"],
                )
                config.output_dir = os.path.join(args.output_dir, exp_key, f"seed_{seed}")
                os.makedirs(config.output_dir, exist_ok=True)

                try:
                    info_list = run_single(config)
                    seed_runs.append(info_list)
                    dt = time.perf_counter() - t0
                    print(f"OK ({dt:.1f}s, {len(info_list)} ticks)", flush=True)
                except Exception as e:
                    print(f"FAILED: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            if seed_runs:
                all_data[exp_key] = aggregate_seeds(seed_runs, n_iters)

    # Save aggregated data
    for key, d in all_data.items():
        save_path = os.path.join(args.output_dir, f"{key}_agg.npz")
        np.savez(save_path, **d)
        print(f"Saved: {save_path}")

    # Plot
    print("\nGenerating plots...")
    plot_results(all_data, args.output_dir, ref_qpos, ctrl_steps, nq, nq_obj)
    print(f"\n✅ All done. Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
