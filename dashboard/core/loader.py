"""
core/loader.py — SPIDER Dashboard Data Loader

Loads trajectory npz files into typed TrajectoryData dataclasses.
No dependency on the spider module; reads npz files directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# TrajectoryData dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryData:
    """
    Typed container for a single MJWP trajectory.

    All arrays use the convention::

        n_ticks     — number of optimization ticks in the trajectory
        ctrl_steps  — controller steps per tick (rollout length)
        nq          — number of joint position DOFs
        nu          — number of control DOFs
        max_iter    — maximum MPPI iterations per tick
        n_cost      — number of cost components in trace_cost
    """

    # --- Core state arrays ---
    qpos: np.ndarray          # (n_ticks, ctrl_steps, nq)
    ctrl: np.ndarray          # (n_ticks, ctrl_steps, nu)

    # --- Reward statistics per tick × iteration ---
    rew_mean: np.ndarray      # (n_ticks, max_iter)
    rew_min: np.ndarray       # (n_ticks, max_iter)
    rew_max: np.ndarray       # (n_ticks, max_iter)

    # --- Optimization convergence ---
    opt_steps: np.ndarray     # (n_ticks, 1)   — int64, steps until convergence

    # --- Cost component traces ---
    trace_cost: np.ndarray    # (n_ticks, max_iter, n_cost)

    # --- Joint-position distance to reference ---
    qpos_dist_mean: np.ndarray  # (n_ticks, max_iter)
    qpos_dist_min: np.ndarray   # (n_ticks, max_iter)
    qpos_dist_max: np.ndarray   # (n_ticks, max_iter)

    # --- Reference trajectory (receding-horizon sites) ---
    trace_ref: np.ndarray     # (n_ticks, 1, 1, horizon_steps, n_sites, 3)

    # --- Optional / supplementary arrays (may be None if absent in npz) ---
    qvel: Optional[np.ndarray] = None          # (n_ticks, ctrl_steps, nv)
    time: Optional[np.ndarray] = None          # (n_ticks, ctrl_steps)
    rew_median: Optional[np.ndarray] = None    # (n_ticks, max_iter)
    improvement: Optional[np.ndarray] = None   # (n_ticks, max_iter)
    qpos_dist_median: Optional[np.ndarray] = None
    qvel_dist_mean: Optional[np.ndarray] = None
    qvel_dist_min: Optional[np.ndarray] = None
    qvel_dist_max: Optional[np.ndarray] = None
    qvel_dist_median: Optional[np.ndarray] = None
    qpos_rew_mean: Optional[np.ndarray] = None
    qpos_rew_min: Optional[np.ndarray] = None
    qpos_rew_max: Optional[np.ndarray] = None
    qpos_rew_median: Optional[np.ndarray] = None
    qvel_rew_mean: Optional[np.ndarray] = None
    qvel_rew_min: Optional[np.ndarray] = None
    qvel_rew_max: Optional[np.ndarray] = None
    qvel_rew_median: Optional[np.ndarray] = None

    # --- Inferred metadata ---
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Required keys that must exist in trajectory_mjwp.npz
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: list[str] = [
    "qpos",
    "ctrl",
    "rew_mean",
    "rew_min",
    "rew_max",
    "opt_steps",
    "trace_cost",
    "qpos_dist_mean",
    "qpos_dist_min",
    "qpos_dist_max",
    "trace_ref",
]

_OPTIONAL_KEYS: list[str] = [
    "qvel",
    "time",
    "rew_median",
    "improvement",
    "qpos_dist_median",
    "qvel_dist_mean",
    "qvel_dist_min",
    "qvel_dist_max",
    "qvel_dist_median",
    "qpos_rew_mean",
    "qpos_rew_min",
    "qpos_rew_max",
    "qpos_rew_median",
    "qvel_rew_mean",
    "qvel_rew_min",
    "qvel_rew_max",
    "qvel_rew_median",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_trajectory(output_dir: Path) -> TrajectoryData:
    """
    Load ``trajectory_mjwp.npz`` from *output_dir* and return a
    :class:`TrajectoryData` instance.

    Parameters
    ----------
    output_dir:
        Directory that contains ``trajectory_mjwp.npz``.

    Returns
    -------
    TrajectoryData
        Fully populated dataclass.  Optional fields that are absent in the
        file are set to ``None``; required fields raise ``KeyError`` if
        missing.

    Raises
    ------
    FileNotFoundError
        If ``trajectory_mjwp.npz`` does not exist under *output_dir*.
    KeyError
        If a required array key is absent from the npz file.
    """
    npz_path = Path(output_dir) / "trajectory_mjwp.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"trajectory_mjwp.npz not found in {output_dir}")

    with np.load(npz_path, allow_pickle=False) as npz:
        available = set(npz.files)

        # --- Load required arrays ---
        required: dict[str, np.ndarray] = {}
        for key in _REQUIRED_KEYS:
            if key not in available:
                raise KeyError(
                    f"Required key '{key}' missing from trajectory_mjwp.npz "
                    f"(found: {sorted(available)})"
                )
            required[key] = npz[key]

        # --- Load optional arrays (graceful fallback to None) ---
        optional: dict[str, Optional[np.ndarray]] = {}
        for key in _OPTIONAL_KEYS:
            optional[key] = npz[key] if key in available else None

    # --- Infer metadata from array shapes ---
    qpos = required["qpos"]           # (n_ticks, ctrl_steps, nq)
    ctrl = required["ctrl"]           # (n_ticks, ctrl_steps, nu)
    rew_mean = required["rew_mean"]   # (n_ticks, max_iter)
    trace_cost = required["trace_cost"]  # (n_ticks, max_iter, n_cost)
    trace_ref = required["trace_ref"]    # (n_ticks, 1, 1, horizon, n_sites, 3)

    meta: dict = {
        "n_ticks": qpos.shape[0],
        "ctrl_steps": qpos.shape[1],
        "nq": qpos.shape[2],
        "nu": ctrl.shape[2],
        "max_iter": rew_mean.shape[1],
        "n_cost_components": trace_cost.shape[2],
        "horizon_steps": trace_ref.shape[3] if trace_ref.ndim >= 4 else None,
        "n_sites": trace_ref.shape[4] if trace_ref.ndim >= 5 else None,
    }

    return TrajectoryData(
        # required
        qpos=required["qpos"],
        ctrl=required["ctrl"],
        rew_mean=required["rew_mean"],
        rew_min=required["rew_min"],
        rew_max=required["rew_max"],
        opt_steps=required["opt_steps"],
        trace_cost=required["trace_cost"],
        qpos_dist_mean=required["qpos_dist_mean"],
        qpos_dist_min=required["qpos_dist_min"],
        qpos_dist_max=required["qpos_dist_max"],
        trace_ref=required["trace_ref"],
        # optional
        **optional,
        # meta
        meta=meta,
    )


def load_kinematic_reference(output_dir: Path) -> np.ndarray:
    """
    Load the kinematic reference ``qpos`` from ``trajectory_kinematic.npz``.

    Parameters
    ----------
    output_dir:
        Directory that contains ``trajectory_kinematic.npz``.

    Returns
    -------
    np.ndarray
        The ``qpos`` array from the kinematic trajectory, shape
        ``(n_frames, nq)``.

    Raises
    ------
    FileNotFoundError
        If ``trajectory_kinematic.npz`` is not found.
    KeyError
        If ``qpos`` is not present in the file.
    """
    npz_path = Path(output_dir) / "trajectory_kinematic.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"trajectory_kinematic.npz not found in {output_dir}"
        )

    with np.load(npz_path, allow_pickle=False) as npz:
        if "qpos" not in npz.files:
            raise KeyError(
                f"'qpos' key missing from trajectory_kinematic.npz "
                f"(found: {sorted(npz.files)})"
            )
        return npz["qpos"].copy()
