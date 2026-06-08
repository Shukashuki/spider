"""
tests/test_analyzer.py

Integration tests for core/analyzer.py using real npz data.

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

from core.analyzer import (
    ExperimentMetrics,
    compute_ctrl_discontinuity,
    compute_metrics,
    compute_tick_boundary_discontinuity,
)
from core.loader import TrajectoryData, load_trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    "/home/roy/.openclaw/workspace/spider/example_datasets"
    "/processed/gigahand/xhand/bimanual/p36-tea/0"
)


@pytest.fixture(scope="module")
def traj_data() -> TrajectoryData:
    return load_trajectory(DATA_DIR)


@pytest.fixture(scope="module")
def metrics(traj_data) -> ExperimentMetrics:
    return compute_metrics(traj_data)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_correct_type(metrics):
    assert isinstance(metrics, ExperimentMetrics)


# ---------------------------------------------------------------------------
# opt_steps metrics
# ---------------------------------------------------------------------------


def test_opt_steps_mean_in_valid_range(metrics):
    """opt_steps mean should be between 1 and 32 (max MPPI iterations)."""
    assert 1 <= metrics.opt_steps_mean <= 32, (
        f"opt_steps_mean={metrics.opt_steps_mean} out of expected range [1, 32]"
    )


def test_opt_steps_std_non_negative(metrics):
    assert metrics.opt_steps_std >= 0


def test_opt_steps_per_tick_shape(metrics, traj_data):
    n_ticks = traj_data.qpos.shape[0]
    assert metrics.opt_steps_per_tick.shape == (n_ticks,)


def test_opt_steps_per_tick_values_consistent(metrics):
    """per_tick values should all lie within [1, 32]."""
    assert np.all(metrics.opt_steps_per_tick >= 1), (
        "Some opt_steps values are < 1"
    )
    assert np.all(metrics.opt_steps_per_tick <= 32), (
        "Some opt_steps values are > 32 (max_iter)"
    )


def test_opt_steps_mean_consistent_with_per_tick(metrics):
    """Mean should equal the mean of per_tick array."""
    expected = float(metrics.opt_steps_per_tick.mean())
    assert abs(metrics.opt_steps_mean - expected) < 1e-6


# ---------------------------------------------------------------------------
# Reward metrics
# ---------------------------------------------------------------------------


def test_reward_mean_is_negative(metrics):
    """SPIDER rewards are typically negative or zero (cost minimization).
    
    rew_mean is zero-padded after convergence, so the analyzer must use
    the reward at opt_steps-1 (the last real MPPI iteration).
    """
    assert metrics.reward_mean <= 0, (
        f"reward_mean={metrics.reward_mean} expected to be <= 0"
    )
    # For the real p36-tea dataset, rewards at convergence are truly negative
    assert metrics.reward_mean < 0, (
        f"reward_mean={metrics.reward_mean} expected to be strictly negative "
        f"(check that analyzer uses opt_steps indexing, not rew_mean[:, -1])"
    )


def test_reward_trend_shape(metrics, traj_data):
    n_ticks = traj_data.qpos.shape[0]
    assert metrics.reward_trend.shape == (n_ticks,)


def test_reward_improvement_type(metrics):
    assert isinstance(metrics.reward_improvement, float)


def test_reward_trend_consistent_with_rew_mean(metrics, traj_data):
    """reward_trend[i] should equal rew_mean[i, opt_steps[i]-1]
    (last real MPPI iteration, before zero-padding).
    """
    import numpy as np
    opt_steps = traj_data.opt_steps.squeeze(-1).astype(int)
    n_ticks = traj_data.rew_mean.shape[0]
    max_iter = traj_data.rew_mean.shape[1]
    last_iter = np.clip(opt_steps - 1, 0, max_iter - 1)
    expected = traj_data.rew_mean[np.arange(n_ticks), last_iter].astype(float)
    np.testing.assert_allclose(
        metrics.reward_trend, expected, rtol=1e-5,
        err_msg="reward_trend should be rew_mean[i, opt_steps[i]-1]"
    )


def test_reward_improvement_computed_correctly(metrics):
    """improvement = last_tick_reward - first_tick_reward."""
    expected = float(metrics.reward_trend[-1] - metrics.reward_trend[0])
    assert abs(metrics.reward_improvement - expected) < 1e-6


# ---------------------------------------------------------------------------
# Tick-boundary discontinuity
# ---------------------------------------------------------------------------


def test_qpos_discontinuity_non_negative(metrics):
    assert metrics.qpos_discontinuity_mean >= 0
    assert metrics.qpos_discontinuity_max >= 0


def test_ctrl_discontinuity_non_negative(metrics):
    assert metrics.ctrl_discontinuity_mean >= 0
    assert metrics.ctrl_discontinuity_max >= 0


def test_qpos_discontinuity_max_ge_mean(metrics):
    assert metrics.qpos_discontinuity_max >= metrics.qpos_discontinuity_mean


def test_ctrl_discontinuity_max_ge_mean(metrics):
    assert metrics.ctrl_discontinuity_max >= metrics.ctrl_discontinuity_mean


# ---------------------------------------------------------------------------
# Reward variance
# ---------------------------------------------------------------------------


def test_reward_variance_non_negative(metrics):
    assert metrics.reward_variance >= 0


def test_reward_variance_consistent_with_trend(metrics):
    expected = float(metrics.reward_trend.var())
    assert abs(metrics.reward_variance - expected) < 1e-6


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------


def test_cost_breakdown_is_dict(metrics):
    assert isinstance(metrics.cost_breakdown, dict)


def test_cost_breakdown_correct_number_of_components(metrics, traj_data):
    n_cost = traj_data.trace_cost.shape[2]
    assert len(metrics.cost_breakdown) == n_cost


def test_cost_breakdown_values_are_floats(metrics):
    for key, val in metrics.cost_breakdown.items():
        assert isinstance(val, float), f"cost_breakdown['{key}'] should be float"


def test_cost_breakdown_keys(metrics, traj_data):
    n_cost = traj_data.trace_cost.shape[2]
    expected_keys = {f"cost_{i}" for i in range(n_cost)}
    assert set(metrics.cost_breakdown.keys()) == expected_keys


def test_cost_breakdown_values_consistent_with_trace_cost(metrics, traj_data):
    """Each cost_i should be the mean of trace_cost[:, :, i]."""
    mean_costs = traj_data.trace_cost.mean(axis=(0, 1))
    for i, val in enumerate(mean_costs):
        key = f"cost_{i}"
        assert abs(metrics.cost_breakdown[key] - float(val)) < 1e-5, (
            f"cost_breakdown['{key}'] mismatch"
        )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestComputeTickBoundaryDiscontinuity:
    def test_zero_for_identical_array(self):
        arr = np.ones((5, 10, 7))  # all identical → jump = 0
        mean, max_ = compute_tick_boundary_discontinuity(arr)
        assert mean == pytest.approx(0.0)
        assert max_ == pytest.approx(0.0)

    def test_known_jump_value(self):
        # tick 0 ends at [1, 0, 0], tick 1 starts at [0, 0, 0] → jump = 1
        arr = np.zeros((2, 3, 3))
        arr[0, -1, 0] = 1.0  # last step of tick 0, first dim = 1
        mean, max_ = compute_tick_boundary_discontinuity(arr)
        assert mean == pytest.approx(1.0)
        assert max_ == pytest.approx(1.0)

    def test_single_tick_returns_zero(self):
        arr = np.ones((1, 10, 5))
        mean, max_ = compute_tick_boundary_discontinuity(arr)
        assert mean == 0.0
        assert max_ == 0.0

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_tick_boundary_discontinuity(np.ones((5, 10)))

    def test_real_qpos(self, traj_data):
        mean, max_ = compute_tick_boundary_discontinuity(traj_data.qpos)
        assert mean >= 0
        assert max_ >= mean

    def test_real_ctrl(self, traj_data):
        mean, max_ = compute_ctrl_discontinuity(traj_data.ctrl)
        assert mean >= 0
        assert max_ >= mean
