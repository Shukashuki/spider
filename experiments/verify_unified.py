#!/usr/bin/env python3
"""Verify unified sampling.py: 3 optimizer modes on SPIDER task (tick0, 1 seed).

Runs:
  1. optimizer_mode="dial"  (via optimizer_type="mppi", final_noise_scale=0.1)
  2. optimizer_mode="mppi"  (via optimizer_type="mppi", optimizer_mode="mppi", final_noise_scale=1.0)
  3. optimizer_mode="cma"   (via optimizer_type="mppi_cma")

Compares: convergence curves, final reward, qpos_dist.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spider.config import Config

DEFAULT_TASK = "P0001_4bf4e21a-obj96945373046044"
DEFAULT_DATASET = "hot3d"

MODES = {
    "dial": {
        "label": "DIAL-MPC (unified, mode=dial)",
        "optimizer_type": "mppi",
        "optimizer_mode": "dial",
        "final_noise_scale": 0.1,
        "color": "#4C72B0",
        "extra": {},
    },
    "mppi": {
        "label": "Pure MPPI (unified, mode=mppi)",
        "optimizer_type": "mppi",
        "optimizer_mode": "mppi",
        "final_noise_scale": 1.0,
        "color": "#DD8452",
        "extra": {},
    },
    "cma": {
        "label": "MPPI-CMA (unified, mode=cma)",
        "optimizer_type": "mppi_cma",
        "optimizer_mode": "cma",  # set by run_mjwp routing
        "final_noise_scale": 1.0,
        "color": "#C44E52",
        "extra": {"cma_sigma0": 0.15, "mppi_cma_eta_mu": 0.5, "mppi_cma_eta_sigma": 0.3},
    },
}


def make_config(mode_key, seed, output_dir, max_iters=32):
    m = MODES[mode_key]
    cfg = Config(
        robot_type="xhand",
        embodiment_type="bimanual",
        task=DEFAULT_TASK,
        dataset_name=DEFAULT_DATASET,
        dataset_dir=os.path.join(os.path.dirname(__file__), "..", "example_datasets"),
        data_id=0,
        seed=seed,
        sim_dt=0.01,
        ctrl_dt=0.4,
        horizon=1.6,
        knot_dt=0.4,
        optimizer_type=m["optimizer_type"],
        num_samples=1024,
        temperature=1.0,
        max_num_iterations=max_iters,
        max_sim_steps=40,  # 1 tick only
        improvement_threshold=0.0,
        improvement_check_steps=999,
        contact_guidance=False,
        contact_rew_scale=0.0,
        cma_sigma0=0.15,
        cma_mu_ratio=0.5,
        first_ctrl_noise_scale=0.5,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=m["final_noise_scale"],
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
    # Set optimizer_mode explicitly for mppi mode
    if m["optimizer_mode"] != "dial" and m["optimizer_type"] == "mppi":
        cfg.optimizer_mode = m["optimizer_mode"]
    # Extra config
    for k, v in m["extra"].items():
        setattr(cfg, k, v)
    return cfg


def main():
    from examples.run_mjwp import main as run_mjwp_main

    seed = 0
    max_iters = 100
    out_base = "outputs/verify_unified"
    os.makedirs(out_base, exist_ok=True)

    results = {}

    for mode_key, mode_meta in MODES.items():
        run_dir = os.path.join(out_base, mode_key)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running: {mode_meta['label']}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        cfg = make_config(mode_key, seed, run_dir, max_iters)
        try:
            run_mjwp_main(cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
        dt = time.perf_counter() - t0

        npz_path = os.path.join(run_dir, "trajectory_mjwp.npz")
        if os.path.exists(npz_path):
            data = dict(np.load(npz_path, allow_pickle=True))
            results[mode_key] = data
            rew_u0 = data["rew_u0"]
            n_ticks, n_iters = rew_u0.shape
            opt_steps = int(data["opt_steps"][0])
            final_rew = rew_u0[0, opt_steps - 1]
            print(f"  OK ({dt:.1f}s) ticks={n_ticks} opt_steps={opt_steps} final_rew={final_rew:.4f}")
            if "qpos_dist_mean" in data:
                qd = data["qpos_dist_mean"][0, opt_steps - 1]
                print(f"  qpos_dist={qd:.4f}")
        else:
            print(f"  No trajectory saved")

    if not results:
        print("No results to plot!")
        return

    # ── Plot: convergence curves (tick 0) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: rew_u0 convergence
    ax = axes[0]
    for mode_key, data in results.items():
        meta = MODES[mode_key]
        rew = data["rew_u0"][0, :]  # tick 0
        opt_steps = int(data["opt_steps"][0])
        iters = np.arange(1, opt_steps + 1)
        ax.plot(iters, rew[:opt_steps], label=meta["label"], color=meta["color"], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("rew_u0 (exploit reward)")
    ax.set_title("Tick 0: Exploit Reward Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: qpos_dist convergence
    ax = axes[1]
    for mode_key, data in results.items():
        meta = MODES[mode_key]
        if "qpos_dist_mean" in data:
            qd = data["qpos_dist_mean"][0, :]
            opt_steps = int(data["opt_steps"][0])
            iters = np.arange(1, opt_steps + 1)
            ax.plot(iters, qd[:opt_steps], label=meta["label"], color=meta["color"], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("qpos_dist_mean")
    ax.set_title("Tick 0: Object Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Unified sampling.py Verification — seed={seed}, {max_iters} iters, N=1024", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_base, "verify_unified.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{'Mode':<35} {'Final rew_u0':>12} {'Final qpos':>12} {'opt_steps':>10}")
    print(f"{'-'*70}")
    for mode_key, data in results.items():
        meta = MODES[mode_key]
        opt_steps = int(data["opt_steps"][0])
        fr = data["rew_u0"][0, opt_steps - 1]
        qd = data["qpos_dist_mean"][0, opt_steps - 1] if "qpos_dist_mean" in data else float("nan")
        print(f"{meta['label']:<35} {fr:>12.4f} {qd:>12.4f} {opt_steps:>10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
