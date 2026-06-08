#!/usr/bin/env python3
"""Three-way comparison v4: Shared warmup + rew_exploit (mean trajectory reward).

Records the actual reward of the mean/exploit trajectory at each iteration,
ensuring all optimizers start from the same point and are directly comparable.

v4b: Warmup uses reference init (stable simulation), but focus tick optimization
starts from ZERO controls — tests each optimizer's ability to find good controls
from scratch.

Conditions:
  1. mppi           — vanilla MPPI
  2. mppi_cma_full  — MPPI softmax weighting + full covariance adaptation
  3. cma_es_full    — True CMA-ES: rank-based selection + full covariance
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
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*ccd_iterations.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config, process_config, compute_noise_schedule
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.optimizers.mppi_cma_full import (
    make_optimize_once_fn_mppi_cma_full,
)
from spider.optimizers.cma_full import (
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
from spider.interp import interp


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


def eval_single_trajectory(config, env, rollout, ctrls, ref_slice):
    """Rollout a single trajectory (the mean/exploit) and return its reward."""
    # Expand to (num_samples, H, nu) — all worlds run the same trajectory
    # so step_env gets the correct (N, nu) shape matching env.num_worlds
    N = config.num_samples
    ctrls_expanded = ctrls.unsqueeze(0).expand(N, -1, -1)  # (N, H, nu)

    # Save and restore state so eval doesn't mutate env
    state = save_state(env)
    _, rews, _, _ = rollout(config, env, ctrls_expanded, ref_slice, {})
    load_state(env, state)

    return float(rews[0].cpu().numpy())


def run_shared_warmup(config, focus_ticks, full_iters):
    """Run warmup once with plain MPPI, save snapshots at each focus tick."""
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

    from spider.optimizers.sampling import make_optimize_fn
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
            env_state = save_state(env)
            snapshots.append({
                "tick": tick_idx,
                "env_state": env_state,
                "mj_qpos": mj_data.qpos.copy(),
                "mj_qvel": mj_data.qvel.copy(),
                "mj_time": mj_data.time,
                "ctrls": ctrls.clone(),
                "sim_step": sim_step,
            })

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


def _knots_from_ctrls(ctrls, config):
    num_knots = int(round(config.horizon / config.knot_dt))
    indices = torch.arange(num_knots, device=ctrls.device) * config.knot_steps
    indices = indices.clamp(max=ctrls.shape[0] - 1).long()
    return ctrls[indices]


def run_optimizer_from_snapshot(config, env, ref_data, rollout, ctrl_ref,
                                snapshot, engine, full_iters, zero_init=True):
    """Load snapshot, run optimizer iteration-by-iteration, record rew_exploit each step."""
    tick = snapshot["tick"]
    sim_step = snapshot["sim_step"]

    # Restore env state
    load_state(env, snapshot["env_state"])

    if zero_init:
        # Start optimization from zero controls (tests optimizer's search ability)
        ctrls = torch.zeros_like(snapshot["ctrls"])
    else:
        ctrls = snapshot["ctrls"].clone()

    ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)

    # Temporarily set num_samples to 1 for exploit eval — no, we use eval_single_trajectory
    # We need a separate rollout with num_samples=1. Instead, use the helper.

    # But eval_single_trajectory needs rollout that accepts (1, H, nu).
    # The existing rollout expects (num_samples, H, nu). We need to handle this.
    # Actually, rollout just uses ctrls.shape[0] as N, so passing (1, H, nu) should work.

    # Record rew_exploit for initial ctrls (before any optimization)
    rew_exploit_list = []

    # Eval initial ctrls
    init_rew = eval_single_trajectory(config, env, rollout, ctrls, ref_slice)
    rew_exploit_list.append(init_rew)

    # Build optimizer (single-step)
    if engine == "mppi_cma_full":
        optimize_once = make_optimize_once_fn_mppi_cma_full(rollout)
    elif engine == "cma_es_full":
        optimize_once = make_optimize_once_fn_cma_full(rollout)
    else:
        optimize_once = make_optimize_once_fn(rollout)

    # Noise annealing schedule
    sample_params_list = [
        {"global_noise_scale": config.beta_traj ** i}
        for i in range(full_iters)
    ]

    # CMA state init (for cma variants)
    cma_state = None
    num_knots = int(round(config.horizon / config.knot_dt))
    d = num_knots * config.nu

    if engine == "mppi_cma_full":
        sigma0 = getattr(config, "cma_sigma0", 0.3)
        Sigma_init = (sigma0 ** 2) * torch.eye(d, device=config.device, dtype=torch.float32)
        cma_state = {
            "mean": _knots_from_ctrls(ctrls, config),
            "Sigma": Sigma_init,
            "generation": 0,
        }
        config.mppi_cma_eta_sigma = 0.3
        config.mppi_cma_eta_mu = 0.5
        config.mppi_cma_jitter = 1e-4
    elif engine == "cma_es_full":
        sigma0 = getattr(config, "cma_sigma0", 0.3)
        Sigma_init = (sigma0 ** 2) * torch.eye(d, device=config.device, dtype=torch.float32)
        cma_state = {
            "mean": _knots_from_ctrls(ctrls, config),
            "Sigma": Sigma_init,
            "generation": 0,
        }

    rew_mean_list = []
    rew_max_list = []

    for i in range(full_iters):
        env_params = [{}]

        if engine in ("mppi_cma_full", "cma_es_full"):
            ctrls, terminate, info = optimize_once(
                config, env, ctrls, ref_slice,
                env_params, sample_params_list[i], cma_state,
            )
        else:
            ctrls, terminate, info = optimize_once(
                config, env, ctrls, ref_slice,
                env_params, sample_params_list[i],
            )

        rew_mean_list.append(info["rew_mean"])
        rew_max_list.append(info["rew_max"])

        # Eval the updated mean trajectory
        rew_exploit = eval_single_trajectory(config, env, rollout, ctrls, ref_slice)
        rew_exploit_list.append(rew_exploit)

    return {
        "tick": tick,
        "rew_exploit": np.array(rew_exploit_list),  # length = full_iters + 1 (includes initial)
        "rew_mean": np.array(rew_mean_list),
        "rew_max": np.array(rew_max_list),
        "mj_qpos": snapshot["mj_qpos"],
    }


def aggregate_seeds(all_runs, n_iters):
    n_seeds = len(all_runs)
    n_ticks = min(len(run) for run in all_runs)

    result = {"tick_ids": np.array([all_runs[0][t]["tick"] for t in range(n_ticks)])}

    # rew_exploit has n_iters+1 entries (includes initial)
    for key, length in [("rew_exploit", n_iters + 1), ("rew_mean", n_iters), ("rew_max", n_iters)]:
        arr = np.zeros((n_seeds, n_ticks, length))
        for s, run in enumerate(all_runs):
            for t in range(n_ticks):
                v = run[t][key]
                arr[s, t, :len(v)] = v[:length]
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

    # ── Fig 1: rew_exploit convergence (THE key plot) ──
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
            rew = d["rew_exploit"][:, t_idx, :]  # (n_seeds, n_iters+1)
            m = rew.mean(axis=0)
            s = rew.std(axis=0)
            iters = np.arange(full_iters + 1)  # 0 = initial, 1..64 = after each iter
            ax.plot(iters, m, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=2)
            ax.fill_between(iters, m - s, m + s, color=style["color"], alpha=0.15)
        ax.set_title(f"Tick {focus_tick}", fontsize=12)
        ax.set_xlabel("Iteration (0 = initial)")
        ax.set_ylabel("Mean Trajectory Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mean Trajectory Reward: Zero Init, Shared Warmup", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_exploit.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_exploit.png")

    # ── Fig 2: Bar chart (final exploit reward) ──
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
                    final = d["rew_exploit"][:, t_idx[0], -1]
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
        ax.set_ylabel("Final Exploit Reward")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Final Mean Trajectory Reward", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_bar.png")

    # ── Fig 3: rew_max convergence (for reference) ──
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

    plt.suptitle("Best Sample Reward (for reference)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_rew_max.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_rew_max.png")

    # ── Summary table ──
    print("\n" + "=" * 95)
    print(f"{'Condition':<40} {'Tick':>5} {'Init Exploit':>14} {'Final Exploit (mean±std)':>28}")
    print("-" * 95)
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
            init_rew = d["rew_exploit"][:, t_idx, 0]
            final_rew = d["rew_exploit"][:, t_idx, -1]
            label = STYLES[cond_name]["label"]
            print(f"{label:<40} {focus_tick:>5} {init_rew.mean():>12.4f} {final_rew.mean():>12.4f} ± {final_rew.std():<10.4f}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="p36-tea")
    parser.add_argument("--dataset_name", default="gigahand")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--output_dir", default="outputs/three_way_v4b")
    parser.add_argument("--zero_init", action="store_true", default=True,
                        help="Start focus tick optimization from zero controls")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Three-Way v4b (shared warmup + zero init + rew_exploit)")
    log(f"  {len(CONDITIONS)} conditions × {args.seeds} seeds × {args.iters} iters")
    log(f"  Focus ticks: {FOCUS_TICKS}, samples: {args.num_samples}")

    all_cond_seed_runs = {cond: [] for cond in CONDITIONS}

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
            log(f"  Warmup done ({dt:.1f}s), {len(snapshots)} snapshots")
        except Exception as e:
            log(f"  WARMUP FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        for cond_name, cond_meta in CONDITIONS.items():
            tick_results = []
            for snap in snapshots:
                tick = snap["tick"]
                log(f"  [{cond_name}] tick={tick}, seed={seed} ...")
                t0 = time.perf_counter()

                try:
                    # Restore noise schedule (process_config sets it once)
                    config = compute_noise_schedule(config)

                    result = run_optimizer_from_snapshot(
                        config, env, ref_data, rollout, ctrl_ref,
                        snap, cond_meta["engine"], args.iters)
                    tick_results.append(result)
                    dt = time.perf_counter() - t0
                    init_r = result["rew_exploit"][0]
                    final_r = result["rew_exploit"][-1]
                    log(f"    OK ({dt:.1f}s) init={init_r:.4f} final={final_r:.4f}")
                except Exception as e:
                    log(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            if tick_results:
                all_cond_seed_runs[cond_name].append(tick_results)

    all_data = {}
    for cond_name, seed_runs in all_cond_seed_runs.items():
        if seed_runs:
            all_data[cond_name] = aggregate_seeds(seed_runs, args.iters)
            path = os.path.join(args.output_dir, f"{cond_name}_agg.npz")
            np.savez(path, **all_data[cond_name])
            log(f"Saved: {path}")

    # Verify initial exploit rewards match
    log("\n--- Initial Exploit Reward Verification ---")
    for tick_i, focus_tick in enumerate(FOCUS_TICKS):
        log(f"Tick {focus_tick}:")
        for cond_name in CONDITIONS:
            if cond_name in all_data:
                d = all_data[cond_name]
                init_rew = d["rew_exploit"][:, tick_i, 0]
                log(f"  {cond_name:20s} init_exploit: {init_rew}")

    log("\nGenerating plots...")
    plot_results(all_data, args.output_dir, args.iters)
    log(f"\n✅ Done. Results in: {args.output_dir}/")
    log_f.close()


if __name__ == "__main__":
    main()
