#!/usr/bin/env python3
"""Closed-loop comparison: MPPI vs MPPI+CMA vs CMA-ES (Hansen).

Runs full closed-loop trajectories with each optimizer, multiple seeds.
Uses SPIDER's run_mjwp.main() directly — same pipeline as production runs.

Phase 1: Single task, multiple seeds (variance estimation)
Phase 2: Multiple tasks (generalization) — extend TASKS list

Usage:
    cd /path/to/spider
    python -u experiments/run_closed_loop_compare.py [--seeds 5] [--task P0001...] [--output_dir outputs/closed_loop_v1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import copy
from dataclasses import fields
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config


# ── Optimizer conditions ──
OPTIMIZERS = {
    "mppi_pure": {
        "label": "Pure MPPI (no annealing)",
        "optimizer_type": "mppi",
        "final_noise_scale": 1.0,
        "color": "#DD8452",
    },
    "dial_mpc": {
        "label": "DIAL-MPC (annealing)",
        "optimizer_type": "mppi",
        "final_noise_scale": 0.1,
        "color": "#4C72B0",
    },
    "mppi_cma": {
        "label": "MPPI+CMA (softmax)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#55A868",
        "extra_config": {"mppi_cma_mean_update": "mppi"},
    },
    "mppi_cma_rank": {
        "label": "MPPI+CMA (rank, anneal)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#C44E52",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5},
    },
    "mppi_cma_rank_10": {
        "label": "MPPI+CMA (rank, 10%)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#8B0000",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.1},
    },
    "mppi_cma_rank_noanneal": {
        "label": "MPPI+CMA (rank, no anneal)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#FF6347",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5},
    },
    "mppi_cma_rank_noanneal_eta07": {
        "label": "MPPI+CMA (rank, no anneal, ηΣ=0.7)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#FF4500",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5, "mppi_cma_eta_sigma": 0.7},
    },
    "mppi_cma_rank_noanneal_s005": {
        "label": "MPPI+CMA (no anneal, σ₀=0.05)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#DAA520",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5, "cma_sigma0": 0.05},
    },
    "mppi_cma_noanneal_eta02": {
        "label": "no anneal, η_μ=0.2",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#8B4513",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5, "cma_sigma0": 0.05, "mppi_cma_eta_mu": 0.2},
    },
    "mppi_cma_noanneal_eta01": {
        "label": "no anneal, η_μ=0.1",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#2E8B57",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5, "cma_sigma0": 0.05, "mppi_cma_eta_mu": 0.1},
    },
    "mppi_cma_noanneal_eta005": {
        "label": "no anneal, η_μ=0.05",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 1.0,
        "color": "#4169E1",
        "extra_config": {"mppi_cma_mean_update": "rank", "cma_mu_ratio": 0.5, "cma_sigma0": 0.05, "mppi_cma_eta_mu": 0.05},
    },
    "mppi_cma_lowT": {
        "label": "MPPI+CMA (T=0.01)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#8172B2",
        "extra_config": {"mppi_cma_mean_update": "mppi", "temperature": 0.01},
    },
    "mppi_cma_T0.1": {
        "label": "MPPI+CMA (T=0.1)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#937860",
        "extra_config": {"mppi_cma_mean_update": "mppi", "temperature": 0.1},
    },
    "mppi_cma_T0.5": {
        "label": "MPPI+CMA (T=0.5)",
        "optimizer_type": "mppi_cma",
        "final_noise_scale": 0.1,
        "color": "#DA8BC3",
        "extra_config": {"mppi_cma_mean_update": "mppi", "temperature": 0.5},
    },
    "cma_full_50": {
        "label": "CMA-ES full (50%)",
        "optimizer_type": "cma_full",
        "final_noise_scale": 0.1,
        "color": "#2CA02C",
        "extra_config": {"cma_mu_ratio": 0.5},
    },
    "cma_full_10": {
        "label": "CMA-ES full (10%)",
        "optimizer_type": "cma_full",
        "final_noise_scale": 0.1,
        "color": "#006400",
        "extra_config": {"cma_mu_ratio": 0.1},
    },
}

# ── Default task (Phase 1) ──
DEFAULT_TASK = "P0001_4bf4e21a-obj96945373046044"
DEFAULT_DATASET = "hot3d"


def make_config(task, dataset_name, seed, optimizer_type, output_dir,
                num_samples=1024, horizon=1.6, max_iters=32,
                temperature=1.0, sigma0=0.15, final_noise_scale=0.1,
                max_sim_steps=-1, extra_config=None) -> Config:
    cfg = Config(
        robot_type="xhand",
        embodiment_type="bimanual",
        task=task,
        dataset_name=dataset_name,
        dataset_dir=os.path.join(os.path.dirname(__file__), "..", "example_datasets"),
        data_id=0,
        seed=seed,
        sim_dt=0.01,
        ctrl_dt=0.4,
        horizon=horizon,
        knot_dt=0.4,
        optimizer_type=optimizer_type,
        num_samples=num_samples,
        temperature=temperature,
        max_num_iterations=max_iters,
        max_sim_steps=max_sim_steps,
        improvement_threshold=0.0,
        improvement_check_steps=999,
        contact_guidance=False,
        contact_rew_scale=0.0,
        cma_sigma0=sigma0,
        cma_mu_ratio=0.5,
        first_ctrl_noise_scale=0.5,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=final_noise_scale,
        show_viewer=False,
        viewer="",
        save_video=False,
        save_info=True,
        save_config=True,
        save_rerun=False,
        save_metrics=False,
        use_torch_compile=False,
        init_ctrl_mode="reference",
        output_dir=output_dir,
    )
    # Apply extra config overrides
    if extra_config:
        for k, v in extra_config.items():
            setattr(cfg, k, v)
    return cfg


def load_trajectory(npz_path):
    """Load trajectory_mjwp.npz and extract key metrics."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.keys()}


def plot_comparison(all_results, output_dir, n_seeds):
    """Plot closed-loop trajectory metrics across optimizers."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Fig 1: rew_u0 (exploit) per tick ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    metrics = [
        ("rew_u0", "Exploit Reward (rew_u0) per Tick"),
        ("rew_max", "Max Reward per Tick"),
        ("rew_mean", "Mean Reward per Tick"),
    ]

    for col, (metric, title) in enumerate(metrics):
        ax = axes[0, col]
        for opt_name, opt_meta in OPTIMIZERS.items():
            if opt_name not in all_results:
                continue
            seeds_data = all_results[opt_name]
            n_ticks = min(d[metric].shape[0] for d in seeds_data)
            # For each tick, take the LAST VALID iteration value (respecting opt_steps)
            final_vals = np.array([
                [d[metric][t, int(d["opt_steps"][t].item()) - 1]
                 for t in range(n_ticks)]
                for d in seeds_data
            ])  # (n_seeds, n_ticks)
            m = final_vals.mean(axis=0)
            s = final_vals.std(axis=0)
            ticks = np.arange(n_ticks)
            ax.plot(ticks, m, color=opt_meta["color"], label=opt_meta["label"], linewidth=2)
            ax.fill_between(ticks, m - s, m + s, color=opt_meta["color"], alpha=0.15)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Tick")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Closed-Loop Trajectory Comparison ({n_seeds} seeds)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_per_tick.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_per_tick.png")

    # ── Fig 2: Convergence curves for selected ticks (rew_u0) ──
    # Pick 3 ticks: early, mid, late
    sample_data = list(all_results.values())[0][0]
    n_ticks = sample_data["rew_u0"].shape[0]
    n_iters = sample_data["rew_u0"].shape[1]
    focus_ticks = [
        max(0, n_ticks // 4),
        n_ticks // 2,
        min(n_ticks - 1, 3 * n_ticks // 4),
    ]

    fig, axes = plt.subplots(1, len(focus_ticks), figsize=(6 * len(focus_ticks), 5),
                             squeeze=False)
    for col, tick in enumerate(focus_ticks):
        ax = axes[0, col]
        for opt_name, opt_meta in OPTIMIZERS.items():
            if opt_name not in all_results:
                continue
            seeds_data = all_results[opt_name]
            # rew_u0 convergence at this tick across seeds
            # Mask out zero-padded iters beyond opt_steps
            curves = []
            for d in seeds_data:
                row = d["rew_u0"][tick, :].copy().astype(float)
                n_valid = int(d["opt_steps"][tick].item())
                row[n_valid:] = np.nan
                curves.append(row)
            curves = np.array(curves)  # (n_seeds, n_iters)
            m = np.nanmean(curves, axis=0)
            s = np.nanstd(curves, axis=0)
            iters = np.arange(n_iters)
            # Only plot where we have valid data
            valid = ~np.isnan(m)
            ax.plot(iters[valid], m[valid], color=opt_meta["color"],
                    label=opt_meta["label"], linewidth=2)
            ax.fill_between(iters[valid], (m - s)[valid], (m + s)[valid],
                            color=opt_meta["color"], alpha=0.15)
        ax.set_title(f"Tick {tick}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("rew_u0 (exploit)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Within-Tick Convergence — rew_u0 (Closed-Loop)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_convergence.png")

    # ── Fig 3: Object tracking error (qpos distance) ──
    if "qpos_dist_mean" in sample_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        for opt_name, opt_meta in OPTIMIZERS.items():
            if opt_name not in all_results:
                continue
            seeds_data = all_results[opt_name]
            # qpos_dist_mean: (n_ticks, n_iters), take last VALID iter per tick
            n_ticks_local = min(d["qpos_dist_mean"].shape[0] for d in seeds_data)
            final_dist = np.array([
                [d["qpos_dist_mean"][t, int(d["opt_steps"][t].item()) - 1]
                 for t in range(n_ticks_local)]
                for d in seeds_data
            ])
            m = final_dist.mean(axis=0)
            s = final_dist.std(axis=0)
            ticks = np.arange(n_ticks_local)
            ax.plot(ticks, m, color=opt_meta["color"], label=opt_meta["label"], linewidth=2)
            ax.fill_between(ticks, m - s, m + s, color=opt_meta["color"], alpha=0.15)
        ax.set_title("Object Tracking Error (qpos distance)", fontsize=12)
        ax.set_xlabel("Tick")
        ax.set_ylabel("qpos_dist_mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig3_tracking.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fig3_tracking.png")

    # ── Summary table ──
    print("\n" + "=" * 90)
    print(f"{'Optimizer':<30} {'Avg Final Rew':>15} {'Avg qpos_dist':>15} {'Avg opt_steps':>15}")
    print("-" * 90)
    for opt_name, opt_meta in OPTIMIZERS.items():
        if opt_name not in all_results:
            continue
        seeds_data = all_results[opt_name]
        n_ticks_local = min(d["rew_u0"].shape[0] for d in seeds_data)
        final_rews = np.array([
            np.mean([d["rew_u0"][t, int(d["opt_steps"][t].item()) - 1]
                     for t in range(n_ticks_local)])
            for d in seeds_data
        ])
        if "qpos_dist_mean" in seeds_data[0]:
            final_dists = np.array([
                np.mean([d["qpos_dist_mean"][t, int(d["opt_steps"][t].item()) - 1]
                         for t in range(n_ticks_local)])
                for d in seeds_data
            ])
            dist_str = f"{final_dists.mean():.4f} ± {final_dists.std():.4f}"
        else:
            dist_str = "N/A"
        avg_steps = np.array([d["opt_steps"].mean() for d in seeds_data])
        print(f"{opt_meta['label']:<30} {final_rews.mean():.4f} ± {final_rews.std():.4f}   {dist_str:>15}   {avg_steps.mean():.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--max_iters", type=int, default=32)
    parser.add_argument("--horizon", type=float, default=1.6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma0", type=float, default=0.15)
    parser.add_argument("--output_dir", default="outputs/closed_loop_v1")
    parser.add_argument("--optimizers", nargs="+",
                        default=list(OPTIMIZERS.keys()),
                        choices=list(OPTIMIZERS.keys()))
    parser.add_argument("--max_ticks", type=int, default=-1,
                        help="Limit number of MPC ticks (-1 = full trajectory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Closed-Loop Comparison v1")
    log(f"  Optimizers: {args.optimizers}")
    log(f"  {args.seeds} seeds × {args.max_iters} iters × {args.num_samples} samples")
    log(f"  Task: {args.task} ({args.dataset_name})")
    log(f"  H={args.horizon}, T={args.temperature}, σ0={args.sigma0}")

    # Import run_mjwp.main here (after sys.path is set)
    from examples.run_mjwp import main as run_mjwp_main

    all_results = {}

    for opt_name in args.optimizers:
        opt_meta = OPTIMIZERS[opt_name]
        seed_results = []

        for seed in range(args.seeds):
            run_dir = os.path.join(args.output_dir, opt_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            log(f"\n  [{opt_meta['label']}] seed={seed} ...")
            t0 = time.perf_counter()

            try:
                # Compute max_sim_steps from max_ticks
                ctrl_steps = int(round(0.4 / 0.01))  # ctrl_dt / sim_dt = 40
                max_sim_steps = args.max_ticks * ctrl_steps if args.max_ticks > 0 else -1

                config = make_config(
                    args.task, args.dataset_name, seed,
                    opt_meta["optimizer_type"], run_dir,
                    num_samples=args.num_samples,
                    horizon=args.horizon,
                    max_iters=args.max_iters,
                    temperature=args.temperature,
                    sigma0=args.sigma0,
                    final_noise_scale=opt_meta.get("final_noise_scale", 0.1),
                    max_sim_steps=max_sim_steps,
                    extra_config=opt_meta.get("extra_config"),
                )

                errors = run_mjwp_main(config)
                dt = time.perf_counter() - t0

                # Load saved trajectory
                npz_path = os.path.join(run_dir, "trajectory_mjwp.npz")
                if os.path.exists(npz_path):
                    traj = load_trajectory(npz_path)
                    seed_results.append(traj)
                    n_ticks = traj["rew_mean"].shape[0]
                    final_rew = traj["rew_mean"][-1, -1] if traj["rew_mean"].ndim == 2 else traj["rew_mean"][-1]
                    log(f"    OK ({dt:.1f}s) ticks={n_ticks} final_rew={final_rew:.4f}")
                    if errors:
                        log(f"    obj_pos_err={errors['obj_pos_err']:.4f} obj_quat_err={errors['obj_quat_err']:.4f}")
                else:
                    log(f"    OK ({dt:.1f}s) but no trajectory saved")

            except Exception as e:
                dt = time.perf_counter() - t0
                log(f"    FAILED ({dt:.1f}s): {e}")
                import traceback
                traceback.print_exc()

        if seed_results:
            all_results[opt_name] = seed_results
            # Save aggregated npz
            agg_path = os.path.join(args.output_dir, f"{opt_name}_seeds.npz")
            # Just save individual seed paths for now
            log(f"  {opt_meta['label']}: {len(seed_results)}/{args.seeds} seeds completed")

    log("\nGenerating plots...")
    try:
        plot_comparison(all_results, args.output_dir, args.seeds)
    except Exception as e:
        log(f"  Plot failed: {e}")
        import traceback
        traceback.print_exc()

    log(f"\n✅ Done. Results in: {args.output_dir}/")
    log_f.close()


if __name__ == "__main__":
    main()
