"""Benchmark: MPPI vs CMA vs MPPI-CMA optimiser speed & quality.

Runs a short window (ticks 4-6, ~0.3s real, 0.1s per tick at ctrl_dt=0.1)
with 64 iterations, 10 seeds, comparing optimizer types under different
gain-decay / contact-cost conditions.

Usage:
    cd spider/
    .venv/bin/python experiments/bench_optimizer_speed.py
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
RUN_SCRIPT = BASE / "examples" / "run_mjwp.py"
PYTHON = BASE / ".venv" / "bin" / "python"
OUTPUT_ROOT = BASE / "outputs" / "bench_optimizer_speed" / "p36-tea"

TASK = "p36-tea"
DATASET = "gigahand"
NUM_SEEDS = 10
MAX_ITERATIONS = 64

# We want ticks 4-6 only.  ctrl_dt=0.2 from gigahand_act → each tick = 0.2s.
# Tick 4 starts at sim_step = 4 * ctrl_steps.  With sim_dt=0.005, ctrl_steps=40.
# To run only ticks 4-6 we set max_sim_steps = 7 * ctrl_steps = 280 (stop after tick 6).
# The script will run from tick 0 but we only analyse ticks 4-6 in post-processing.
# Actually, to save time, let's just run 7 ticks total (0-6) and filter later.
MAX_SIM_STEPS = 280  # 7 ticks × 40 steps/tick

OPTIMIZERS = ["mppi", "cma", "mppi_cma"]

CONDITIONS = {
    "gain_decay": {
        "contact_guidance": "true",
        "contact_rew_scale": "0.0",
    },
    "no_guidance": {
        "contact_guidance": "false",
        "contact_rew_scale": "0.0",
    },
    "decay_plus_cost": {
        "contact_guidance": "true",
        "contact_rew_scale": "1.0",
    },
    "cost_only": {
        "contact_guidance": "false",
        "contact_rew_scale": "1.0",
    },
}


def run_one(opt_type: str, cond_name: str, cond_overrides: dict, seed: int) -> dict:
    """Run a single experiment and return timing + reward info."""
    tag = f"{opt_type}__{cond_name}__seed{seed}"
    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keys present in default.yaml can be overridden directly.
    # Keys only in gigahand_act.yaml (not in default.yaml) need '+' prefix for Hydra.
    cmd = [
        str(PYTHON), str(RUN_SCRIPT),
        f"task={TASK}",
        f"dataset_name={DATASET}",
        "robot_type=xhand",
        "embodiment_type=bimanual",
        "data_id=0",
        "simulator=mjwp",
        "device=cuda:0",
        f"output_dir={out_dir}",
        "show_viewer=false",
        "save_video=false",
        "save_info=true",
        "save_metrics=true",
        "save_config=true",
        f"seed={seed}",
        f"optimizer_type={opt_type}",
        f"max_num_iterations={MAX_ITERATIONS}",
        f"max_sim_steps={MAX_SIM_STEPS}",
        # Keys in default.yaml
        "temperature=1.0",
        "num_samples=1024",
        "first_ctrl_noise_scale=1.0",
        "final_noise_scale=0.3",
        "sim_dt=0.005",
        "gibbs_sampling=false",
        "improvement_threshold=0.0",
        "joint_noise_scale=0.15",
        "pos_noise_scale=0.01",
        "rot_noise_scale=0.01",
        "horizon=1.6",
        "knot_dt=0.4",
        "ctrl_dt=0.2",
        "base_pos_rew_scale=0.03",
        "base_rot_rew_scale=0.03",
        "pos_rew_scale=1.0",
        "rot_rew_scale=1.0",
        # Keys NOT in default.yaml → need '+' prefix
        "+guidance_decay_ratio=0.5",
        "+init_pos_actuator_gain=2.0",
        "+init_pos_actuator_bias=2.0",
        "+init_rot_actuator_gain=0.3",
        "+init_rot_actuator_bias=0.3",
    ]
    for k, v in cond_overrides.items():
        cmd.append(f"{k}={v}")

    print(f"\n{'='*70}")
    print(f"  [{tag}]  optimizer={opt_type}  cond={cond_name}  seed={seed}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    wall_time = time.perf_counter() - t0

    status = "ok" if result.returncode == 0 else "fail"
    if result.returncode != 0:
        print(f"  ⚠️  exit={result.returncode}")
        # Print last 20 lines of stderr for debugging
        err_lines = result.stderr.strip().split("\n")[-20:]
        for line in err_lines:
            print(f"    {line}")

    # Try to load saved trajectory for ticks 4-6 analysis
    tick_rewards = None
    opt_steps_per_tick = None
    npz_candidates = list(out_dir.glob("trajectory_mjwp*.npz"))
    if npz_candidates:
        data = np.load(npz_candidates[0], allow_pickle=True)
        # Each tick is one entry in the stacked info_list
        # We want ticks 4, 5, 6 (0-indexed)
        if "rew_mean" in data and len(data["rew_mean"]) > 6:
            tick_rewards = {
                "rew_mean": data["rew_mean"][4:7].tolist(),
                "rew_max": data["rew_max"][4:7].tolist(),
                "rew_min": data["rew_min"][4:7].tolist(),
            }
        if "opt_steps" in data and len(data["opt_steps"]) > 6:
            opt_steps_per_tick = data["opt_steps"][4:7].tolist()

    record = {
        "tag": tag,
        "optimizer": opt_type,
        "condition": cond_name,
        "seed": seed,
        "wall_time_s": round(wall_time, 2),
        "status": status,
        "tick_rewards": tick_rewards,
        "opt_steps_per_tick": opt_steps_per_tick,
    }
    print(f"  ✅ wall={wall_time:.1f}s  status={status}")
    return record


def main():
    all_records = []
    combos = list(itertools.product(OPTIMIZERS, CONDITIONS.items(), range(NUM_SEEDS)))
    total = len(combos)

    print(f"Total runs: {total}  ({len(OPTIMIZERS)} optimizers × "
          f"{len(CONDITIONS)} conditions × {NUM_SEEDS} seeds)")
    print(f"Output: {OUTPUT_ROOT}\n")

    for i, (opt, (cond_name, cond_ov), seed) in enumerate(combos):
        print(f"\n[{i+1}/{total}]", end="")
        record = run_one(opt, cond_name, cond_ov, seed)
        all_records.append(record)

        # Save incrementally
        summary_path = OUTPUT_ROOT / "bench_results.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_records, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Optimizer':<12} {'Condition':<18} {'Wall(s)':<10} {'OptSteps':<12} {'RewMean':<12} {'Status'}")
    print("-" * 80)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_records:
        key = (r["optimizer"], r["condition"])
        grouped[key].append(r)

    for (opt, cond), records in sorted(grouped.items()):
        walls = [r["wall_time_s"] for r in records]
        statuses = [r["status"] for r in records]
        ok_count = sum(1 for s in statuses if s == "ok")

        # Average tick 4-6 reward across seeds
        rew_means = []
        opt_steps_all = []
        for r in records:
            if r["tick_rewards"] and r["tick_rewards"]["rew_mean"]:
                rew_means.extend(r["tick_rewards"]["rew_mean"])
            if r["opt_steps_per_tick"]:
                opt_steps_all.extend(r["opt_steps_per_tick"])

        avg_wall = np.mean(walls)
        avg_rew = np.mean(rew_means) if rew_means else float("nan")
        avg_opt = np.mean(opt_steps_all) if opt_steps_all else float("nan")

        print(f"{opt:<12} {cond:<18} {avg_wall:<10.1f} {avg_opt:<12.1f} {avg_rew:<12.4f} {ok_count}/{len(records)} ok")

    print(f"\nResults saved to: {OUTPUT_ROOT / 'bench_results.json'}")


if __name__ == "__main__":
    main()
