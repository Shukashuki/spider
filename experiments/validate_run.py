#!/usr/bin/env python3
"""Validate a single SPIDER experiment run.

Checks: trajectory file exists, no NaN/Inf, reasonable tracking error range.
Usage:
    python experiments/validate_run.py --exp_dir outputs/ablation/p36-tea/baseline
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def validate(exp_dir: str, verbose: bool = True) -> dict:
    """Validate a single experiment output directory.

    Returns dict with keys: passed (bool), checks (list of (name, passed, msg)).
    """
    checks = []

    # --- Check 1: trajectory file exists ---
    traj_path = None
    for name in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        candidate = os.path.join(exp_dir, name)
        if os.path.exists(candidate):
            traj_path = candidate
            break

    if traj_path is None:
        checks.append(("traj_exists", False, "No trajectory_mjwp*.npz found"))
        return {"passed": False, "checks": checks}

    checks.append(("traj_exists", True, f"Found {os.path.basename(traj_path)}"))

    # --- Load data ---
    try:
        data = np.load(traj_path)
    except Exception as e:
        checks.append(("traj_loadable", False, f"Failed to load: {e}"))
        return {"passed": False, "checks": checks}
    checks.append(("traj_loadable", True, f"Keys: {sorted(data.files)}"))

    # --- Check 2: qpos exists and shape is sane ---
    if "qpos" not in data:
        checks.append(("qpos_exists", False, "No 'qpos' key in npz"))
        return {"passed": False, "checks": checks}

    qpos = data["qpos"]
    checks.append(("qpos_exists", True, f"shape={qpos.shape}, dtype={qpos.dtype}"))

    # --- Check 3: no NaN / Inf ---
    nan_count = int(np.isnan(qpos).sum())
    inf_count = int(np.isinf(qpos).sum())
    if nan_count > 0 or inf_count > 0:
        checks.append(("no_nan_inf", False, f"NaN={nan_count}, Inf={inf_count}"))
    else:
        checks.append(("no_nan_inf", True, "Clean"))

    # --- Check 4: qpos values in reasonable range ---
    qpos_abs_max = float(np.abs(qpos).max())
    if qpos_abs_max > 100.0:
        checks.append(("qpos_range", False, f"abs max={qpos_abs_max:.2f} (>100, likely diverged)"))
    elif qpos_abs_max > 20.0:
        checks.append(("qpos_range", True, f"abs max={qpos_abs_max:.2f} (warning: high)"))
    else:
        checks.append(("qpos_range", True, f"abs max={qpos_abs_max:.2f}"))

    # --- Check 5: reward info ---
    for key in ["qpos_rew", "qvel_rew"]:
        if key in data:
            arr = data[key]
            if np.isnan(arr).any():
                checks.append((f"{key}_valid", False, "Contains NaN"))
            else:
                checks.append((f"{key}_valid", True,
                              f"mean={float(arr.mean()):.4f}, last={float(arr[-1].mean()) if arr.ndim > 1 else float(arr[-1]):.4f}"))

    # --- Check 6: config file saved ---
    config_exists = any(
        os.path.exists(os.path.join(exp_dir, f))
        for f in ["config.yaml", "config_act.yaml"]
    )
    checks.append(("config_saved", config_exists,
                   "Found" if config_exists else "No config yaml"))

    # --- Summary ---
    all_passed = all(c[1] for c in checks)

    if verbose:
        status = "PASS ✅" if all_passed else "FAIL ❌"
        print(f"\n{'='*60}")
        print(f"  Validation: {exp_dir}")
        print(f"  Status: {status}")
        print(f"{'='*60}")
        for name, passed, msg in checks:
            icon = "✓" if passed else "✗"
            print(f"  [{icon}] {name}: {msg}")
        print()

    return {"passed": all_passed, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Validate a SPIDER experiment run")
    parser.add_argument("--exp_dir", required=True, help="Experiment output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    result = validate(args.exp_dir, verbose=not args.quiet)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
