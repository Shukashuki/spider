"""Ablation: MPPI vs MPPI-CMA convergence on p36-tea (3 seeds).

Both conditions use gain decay (contact_guidance=True).
Only difference: optimizer_type.

Usage:
    cd spider/
    .venv/bin/python experiments/ablation_optimizer.py
"""

import subprocess
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SCRIPT = os.path.join(BASE, "examples", "run_mjwp.py")
PYTHON = os.path.join(BASE, ".venv", "bin", "python")

TASK = "p36-tea"
DATASET = "gigahand"
SEEDS = [0, 1, 2]

CONDITIONS = {
    "mppi": {
        "optimizer_type": "mppi",
    },
    "mppi_cma": {
        "optimizer_type": "mppi_cma",
    },
}


def run_condition(name, overrides, seed):
    output_dir = os.path.join(
        BASE, "outputs", "ablation_optimizer", TASK, f"{name}_seed{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        PYTHON, RUN_SCRIPT,
        f"task={TASK}",
        f"dataset_name={DATASET}",
        "robot_type=xhand",
        "embodiment_type=bimanual",
        "data_id=0",
        "simulator=mjwp",
        "device=cuda:0",
        "contact_guidance=true",
        "contact_rew_scale=0.0",
        "horizon=1.6",
        "max_sim_steps=200",
        f"seed={seed}",
        f"output_dir={output_dir}",
        "show_viewer=false",
        "save_video=false",
        "save_info=true",
    ]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")

    print(f"\n{'='*60}")
    print(f"Running: {name} | seed={seed}")
    print(f"  optimizer: {overrides['optimizer_type']}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=BASE, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️  {name}_seed{seed} exited with code {result.returncode}")
    else:
        print(f"✅ {name}_seed{seed} done")
    return result.returncode


def main():
    results = {}
    for name, overrides in CONDITIONS.items():
        for seed in SEEDS:
            key = f"{name}_seed{seed}"
            rc = run_condition(name, overrides, seed)
            results[key] = rc

    print(f"\n{'='*60}")
    print("Summary:")
    for key, rc in results.items():
        status = "✅" if rc == 0 else "❌"
        print(f"  {status} {key} (exit={rc})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
