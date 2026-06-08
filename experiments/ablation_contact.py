"""Ablation: gain decay vs contact cost on p36-tea (3 seeds).

4 conditions × 3 seeds:
  A) original:        contact_guidance=True,  contact_rew_scale=0.0  (baseline)
  B) no_gain_decay:   contact_guidance=False, contact_rew_scale=0.0  (no guidance at all)
  C) contact_cost:    contact_guidance=True,  contact_rew_scale=1.0  (gain decay + contact cost)
  D) cost_only:       contact_guidance=False, contact_rew_scale=1.0  (no gain decay, contact cost only)

Usage:
    cd spider/
    .venv/bin/python experiments/ablation_contact.py
"""

import subprocess
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SCRIPT = os.path.join(BASE, "examples", "run_mjwp.py")
PYTHON = os.path.join(BASE, ".venv", "bin", "python")

TASK = "p36-tea"
DATASET = "gigahand"
SEEDS = [0, 1, 2]

CONDITIONS = {
    "A_original": {
        "contact_guidance": "true",
        "contact_rew_scale": "0.0",
    },
    "B_no_gain_decay": {
        "contact_guidance": "false",
        "contact_rew_scale": "0.0",
    },
    "C_gain_decay_plus_cost": {
        "contact_guidance": "true",
        "contact_rew_scale": "1.0",
    },
    "D_cost_only": {
        "contact_guidance": "false",
        "contact_rew_scale": "1.0",
    },
}


def run_condition(name, overrides, seed):
    output_dir = os.path.join(
        BASE, "outputs", "ablation_contact", TASK, f"{name}_seed{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Hydra uses positional overrides (key=value without --)
    cmd = [
        PYTHON, RUN_SCRIPT,
        f"task={TASK}",
        f"dataset_name={DATASET}",
        "robot_type=xhand",
        "embodiment_type=bimanual",
        "data_id=0",
        "simulator=mjwp",
        "device=cuda:0",
        "optimizer_type=mppi_cma",
        "horizon=0.4",
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
    print(f"  overrides: {overrides}")
    print(f"  output: {output_dir}")
    print(f"  cmd: {' '.join(cmd[-6:])}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=BASE, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️  {name} exited with code {result.returncode}")
    else:
        print(f"✅ {name} done")
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
