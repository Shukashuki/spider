"""
tests/test_loader.py

Integration tests for core/loader.py using real npz data.

Test data directory:
    /home/roy/.openclaw/workspace/spider/example_datasets/
    processed/gigahand/xhand/bimanual/p36-tea/0/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# --- Make sure the dashboard package is importable ---
DASHBOARD_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

from core.loader import TrajectoryData, load_kinematic_reference, load_trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    "/home/roy/.openclaw/workspace/spider/example_datasets"
    "/processed/gigahand/xhand/bimanual/p36-tea/0"
)


@pytest.fixture(scope="module")
def traj_data() -> TrajectoryData:
    """Load real trajectory once for all tests in this module."""
    return load_trajectory(DATA_DIR)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


def test_load_trajectory_returns_correct_type(traj_data):
    assert isinstance(traj_data, TrajectoryData)


def test_load_trajectory_required_arrays_not_none(traj_data):
    """All required fields must be populated (not None)."""
    required_fields = [
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
    for field_name in required_fields:
        value = getattr(traj_data, field_name)
        assert value is not None, f"Field '{field_name}' should not be None"
        assert isinstance(value, np.ndarray), (
            f"Field '{field_name}' should be np.ndarray, got {type(value)}"
        )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_qpos_shape(traj_data):
    # (n_ticks, ctrl_steps, nq)
    assert traj_data.qpos.ndim == 3, "qpos must be 3-D"
    n_ticks, ctrl_steps, nq = traj_data.qpos.shape
    assert n_ticks > 0
    assert ctrl_steps > 0
    assert nq > 0


def test_ctrl_shape_consistent_with_qpos(traj_data):
    n_ticks_q, ctrl_steps_q, _ = traj_data.qpos.shape
    n_ticks_c, ctrl_steps_c, nu = traj_data.ctrl.shape
    assert n_ticks_q == n_ticks_c, "qpos and ctrl must have the same n_ticks"
    assert ctrl_steps_q == ctrl_steps_c, "qpos and ctrl must have the same ctrl_steps"
    assert nu > 0


def test_rew_mean_shape(traj_data):
    # (n_ticks, max_iter)
    assert traj_data.rew_mean.ndim == 2
    n_ticks = traj_data.qpos.shape[0]
    assert traj_data.rew_mean.shape[0] == n_ticks


def test_opt_steps_shape(traj_data):
    # (n_ticks, 1)
    assert traj_data.opt_steps.ndim == 2
    assert traj_data.opt_steps.shape[1] == 1
    n_ticks = traj_data.qpos.shape[0]
    assert traj_data.opt_steps.shape[0] == n_ticks


def test_trace_cost_shape(traj_data):
    # (n_ticks, max_iter, n_cost_components)
    assert traj_data.trace_cost.ndim == 3
    n_ticks = traj_data.qpos.shape[0]
    assert traj_data.trace_cost.shape[0] == n_ticks
    assert traj_data.trace_cost.shape[2] > 0  # at least one cost component


def test_trace_ref_shape(traj_data):
    # (n_ticks, 1, 1, horizon_steps, n_sites, 3)
    assert traj_data.trace_ref.ndim == 6
    assert traj_data.trace_ref.shape[-1] == 3  # x, y, z


# ---------------------------------------------------------------------------
# Metadata (meta dict)
# ---------------------------------------------------------------------------


def test_meta_dict_populated(traj_data):
    meta = traj_data.meta
    assert isinstance(meta, dict)
    for key in ["n_ticks", "ctrl_steps", "nq", "nu", "max_iter", "n_cost_components"]:
        assert key in meta, f"meta should contain '{key}'"
        assert meta[key] is not None and meta[key] > 0


def test_meta_consistent_with_arrays(traj_data):
    meta = traj_data.meta
    assert meta["n_ticks"] == traj_data.qpos.shape[0]
    assert meta["ctrl_steps"] == traj_data.qpos.shape[1]
    assert meta["nq"] == traj_data.qpos.shape[2]
    assert meta["nu"] == traj_data.ctrl.shape[2]
    assert meta["max_iter"] == traj_data.rew_mean.shape[1]
    assert meta["n_cost_components"] == traj_data.trace_cost.shape[2]


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_optional_qvel_type_if_present(traj_data):
    """If qvel is loaded, it must be an ndarray of shape (n_ticks, ctrl_steps, nv)."""
    if traj_data.qvel is not None:
        assert isinstance(traj_data.qvel, np.ndarray)
        assert traj_data.qvel.ndim == 3
        assert traj_data.qvel.shape[0] == traj_data.qpos.shape[0]


def test_optional_time_type_if_present(traj_data):
    if traj_data.time is not None:
        assert isinstance(traj_data.time, np.ndarray)
        assert traj_data.time.ndim == 2
        assert traj_data.time.shape[0] == traj_data.qpos.shape[0]


# ---------------------------------------------------------------------------
# Kinematic reference
# ---------------------------------------------------------------------------


def test_load_kinematic_reference_returns_ndarray():
    ref = load_kinematic_reference(DATA_DIR)
    assert isinstance(ref, np.ndarray)


def test_load_kinematic_reference_shape():
    ref = load_kinematic_reference(DATA_DIR)
    assert ref.ndim == 2, "kinematic qpos should be 2-D (n_frames, nq)"
    n_frames, nq = ref.shape
    assert n_frames > 0
    assert nq > 0


def test_kinematic_reference_nq_matches_trajectory(traj_data):
    """nq of kinematic reference should match nq of mjwp trajectory."""
    ref = load_kinematic_reference(DATA_DIR)
    assert ref.shape[1] == traj_data.qpos.shape[2], (
        f"kinematic nq={ref.shape[1]} != mjwp nq={traj_data.qpos.shape[2]}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_load_trajectory_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_trajectory(Path("/nonexistent/path/to/nowhere"))


def test_load_kinematic_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_kinematic_reference(Path("/nonexistent/path"))
