"""
core/analyzer.py — SPIDER Dashboard Metrics Analyzer

Computes the four key experiment metrics from a TrajectoryData instance.
No dependency on the spider module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.loader import TrajectoryData


# ---------------------------------------------------------------------------
# ExperimentMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """
    Summary metrics for a single MJWP experiment.

    Attributes
    ----------
    opt_steps_mean : float
        Mean convergence steps across all ticks (lower is better).
    opt_steps_std : float
        Standard deviation of convergence steps across ticks.
    opt_steps_per_tick : np.ndarray
        Shape ``(n_ticks,)`` — convergence steps for each tick.

    reward_mean : float
        Mean of the *final* MPPI reward at the last iteration, for the
        last tick.  Negative in typical SPIDER setups (lower cost → less
        negative).
    reward_trend : np.ndarray
        Shape ``(n_ticks,)`` — final-iteration reward for every tick,
        showing how the policy improves over the trajectory.
    reward_improvement : float
        Delta between the *last* tick's final reward and the *first*
        tick's final reward (positive means improvement).

    qpos_discontinuity_mean : float
        Mean L2-norm jump in ``qpos`` at tick boundaries.
    qpos_discontinuity_max : float
        Maximum L2-norm jump in ``qpos`` at tick boundaries.
    ctrl_discontinuity_mean : float
        Mean L2-norm jump in ``ctrl`` at tick boundaries.
    ctrl_discontinuity_max : float
        Maximum L2-norm jump in ``ctrl`` at tick boundaries.

    reward_variance : float
        Variance of the per-tick final-iteration rewards (stability metric).

    cost_breakdown : dict[str, float]
        Mean cost per component, keyed as ``"cost_0"`` … ``"cost_N"``.
        If the npz contains component names, those are used instead.
    """

    # --- Optimization convergence ---
    opt_steps_mean: float
    opt_steps_std: float
    opt_steps_per_tick: np.ndarray    # (n_ticks,)

    # --- Reward quality ---
    reward_mean: float
    reward_trend: np.ndarray          # (n_ticks,)
    reward_improvement: float

    # --- Tick-boundary discontinuity ---
    qpos_discontinuity_mean: float
    qpos_discontinuity_max: float
    ctrl_discontinuity_mean: float
    ctrl_discontinuity_max: float

    # --- Stability ---
    reward_variance: float

    # --- Cost decomposition ---
    cost_breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def compute_tick_boundary_discontinuity(
    arr: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the L2-norm jump between consecutive ticks.

    For tick *i*, the jump is defined as::

        ‖ arr[i, -1, :] − arr[i+1, 0, :] ‖₂

    i.e., the Euclidean distance between the *last* step of tick *i* and
    the *first* step of tick *i+1*.

    Parameters
    ----------
    arr:
        Array of shape ``(n_ticks, steps_per_tick, dim)``.

    Returns
    -------
    mean_jump : float
    max_jump  : float
    """
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (n_ticks, steps, dim), got shape {arr.shape}"
        )
    n_ticks = arr.shape[0]
    if n_ticks < 2:
        return 0.0, 0.0

    # last step of tick i  vs  first step of tick i+1
    last = arr[:-1, -1, :]   # (n_ticks-1, dim)
    first = arr[1:, 0, :]    # (n_ticks-1, dim)
    jumps = np.linalg.norm(last - first, axis=-1)  # (n_ticks-1,)

    return float(jumps.mean()), float(jumps.max())


def compute_ctrl_discontinuity(
    ctrl: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the L2-norm jump in control signal at tick boundaries.

    Same convention as :func:`compute_tick_boundary_discontinuity`.

    Parameters
    ----------
    ctrl:
        Array of shape ``(n_ticks, ctrl_steps, nu)``.

    Returns
    -------
    mean_jump : float
    max_jump  : float
    """
    return compute_tick_boundary_discontinuity(ctrl)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(data: TrajectoryData) -> ExperimentMetrics:
    """
    Compute all four experiment metrics from a :class:`~core.loader.TrajectoryData`.

    Parameters
    ----------
    data:
        Loaded trajectory (output of :func:`~core.loader.load_trajectory`).

    Returns
    -------
    ExperimentMetrics
        Fully populated metrics dataclass.

    Notes
    -----
    * **opt_steps** — ``data.opt_steps`` has shape ``(n_ticks, 1)``.
      We squeeze the trailing dimension before computing statistics.

    * **reward** — We use the *final* MPPI iteration's reward for each
      tick: ``data.rew_mean[:, -1]``.  The "last" iteration is the one
      after the optimizer has fully converged for that tick.

    * **tick boundary discontinuity** — see
      :func:`compute_tick_boundary_discontinuity`.

    * **cost_breakdown** — We average ``data.trace_cost`` over both the
      tick and iteration axes, yielding one scalar per cost component.
    """
    # ------------------------------------------------------------------
    # 1. Optimization convergence
    # ------------------------------------------------------------------
    opt_steps_flat = data.opt_steps.squeeze(-1).astype(float)  # (n_ticks,)
    opt_steps_mean = float(opt_steps_flat.mean())
    opt_steps_std = float(opt_steps_flat.std())

    # ------------------------------------------------------------------
    # 2. Reward quality
    #    rew_mean is zero-padded after convergence.  Use the reward at the
    #    *opt_steps - 1* iteration for each tick (the last real update).
    # ------------------------------------------------------------------
    tick_indices = np.arange(len(opt_steps_flat))
    # opt_steps stores 1-based convergence count; clip to valid range
    last_iter_indices = np.clip(
        data.opt_steps.squeeze(-1).astype(int) - 1,
        0,
        data.rew_mean.shape[1] - 1,
    )
    reward_trend = data.rew_mean[tick_indices, last_iter_indices].astype(float)  # (n_ticks,)
    reward_mean = float(reward_trend.mean())
    reward_improvement = float(reward_trend[-1] - reward_trend[0])

    # ------------------------------------------------------------------
    # 3. Tick-boundary discontinuity
    # ------------------------------------------------------------------
    qpos_disc_mean, qpos_disc_max = compute_tick_boundary_discontinuity(
        data.qpos
    )
    ctrl_disc_mean, ctrl_disc_max = compute_ctrl_discontinuity(data.ctrl)

    # ------------------------------------------------------------------
    # 4. Reward variance (stability)
    # ------------------------------------------------------------------
    reward_variance = float(reward_trend.var())

    # ------------------------------------------------------------------
    # 5. Cost breakdown
    #    trace_cost: (n_ticks, max_iter, n_cost)
    #    Average over ticks and iterations → one scalar per component.
    # ------------------------------------------------------------------
    mean_costs = data.trace_cost.mean(axis=(0, 1))  # (n_cost,)
    cost_breakdown: dict[str, float] = {
        f"cost_{i}": float(mean_costs[i]) for i in range(len(mean_costs))
    }

    return ExperimentMetrics(
        # convergence
        opt_steps_mean=opt_steps_mean,
        opt_steps_std=opt_steps_std,
        opt_steps_per_tick=opt_steps_flat,
        # reward
        reward_mean=reward_mean,
        reward_trend=reward_trend,
        reward_improvement=reward_improvement,
        # discontinuity
        qpos_discontinuity_mean=qpos_disc_mean,
        qpos_discontinuity_max=qpos_disc_max,
        ctrl_discontinuity_mean=ctrl_disc_mean,
        ctrl_discontinuity_max=ctrl_disc_max,
        # variance
        reward_variance=reward_variance,
        # cost
        cost_breakdown=cost_breakdown,
    )
