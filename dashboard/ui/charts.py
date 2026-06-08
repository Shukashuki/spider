"""
ui/charts.py — SPIDER Dashboard Chart Library
==============================================
All chart functions accept numpy arrays and return plotly.graph_objects.Figure.
No Streamlit imports — charts are pure data → Figure transforms.

Color palette (consistent across all charts):
  BLUE   = "#2196F3"
  ORANGE = "#FF9800"
  GREEN  = "#4CAF50"
  RED    = "#F44336"
  PURPLE = "#9C27B0"
  CYAN   = "#00BCD4"

Dark-theme aware: uses plotly "plotly_dark" template as base.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# ── Palette ────────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#00BCD4"]
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN = COLORS

_TEMPLATE = "plotly_dark"

# ── Layout helper ──────────────────────────────────────────────────────────

def _base_layout(**kwargs) -> dict:
    """Return common layout kwargs merged with caller overrides."""
    base = dict(
        template=_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(30,30,30,0.7)", bordercolor="#555", borderwidth=1),
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    base.update(kwargs)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 01 — Reward evolution (viz/01_reward_evolution.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_reward_convergence(
    rew_mean: np.ndarray,
    rew_max: Optional[np.ndarray] = None,
    rew_min: Optional[np.ndarray] = None,
    opt_steps: Optional[np.ndarray] = None,
    selected_ticks: Optional[list] = None,
) -> go.Figure:
    """Plot MPPI reward convergence per selected tick.

    Args:
        rew_mean: shape (n_ticks, n_iters) — mean reward per tick per iteration.
        rew_max: shape (n_ticks, n_iters) — optional max reward.
        rew_min: shape (n_ticks, n_iters) — optional min reward.
        opt_steps: shape (n_ticks,) — actual optimization steps taken per tick.
                   If None, assumes full n_iters for every tick.
        selected_ticks: list of tick indices to plot. Defaults to 4 spread ticks.

    Returns:
        go.Figure with one subplot per selected tick.
    """
    n_ticks, n_iters = rew_mean.shape
    if opt_steps is None:
        opt_steps = np.full(n_ticks, n_iters)
    opt_steps = np.asarray(opt_steps).flatten()

    if selected_ticks is None:
        idxs = np.linspace(0, n_ticks - 1, min(4, n_ticks), dtype=int).tolist()
        selected_ticks = idxs

    n_cols = len(selected_ticks)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=[f"Tick {t}  (steps={int(opt_steps[t])})" for t in selected_ticks],
    )

    for col_i, tick in enumerate(selected_ticks, start=1):
        steps = max(1, int(opt_steps[tick]))
        iters = np.arange(steps)
        show_legend = col_i == 1

        # mean line
        fig.add_trace(
            go.Scatter(
                x=iters, y=rew_mean[tick, :steps],
                mode="lines",
                name="mean",
                line=dict(color=BLUE, width=2),
                legendgroup="mean",
                showlegend=show_legend,
                hovertemplate="iter %{x}<br>mean=%{y:.5f}<extra>mean</extra>",
            ),
            row=1, col=col_i,
        )

        if rew_max is not None:
            fig.add_trace(
                go.Scatter(
                    x=iters, y=rew_max[tick, :steps],
                    mode="lines",
                    name="max",
                    line=dict(color=RED, width=1.5, dash="dot"),
                    legendgroup="max",
                    showlegend=show_legend,
                    hovertemplate="iter %{x}<br>max=%{y:.5f}<extra>max</extra>",
                ),
                row=1, col=col_i,
            )

        if rew_min is not None:
            fig.add_trace(
                go.Scatter(
                    x=iters, y=rew_min[tick, :steps],
                    mode="lines",
                    name="min",
                    line=dict(color=GREEN, width=1.5, dash="dot"),
                    legendgroup="min",
                    showlegend=show_legend,
                    hovertemplate="iter %{x}<br>min=%{y:.5f}<extra>min</extra>",
                ),
                row=1, col=col_i,
            )

        # fill between min and max
        if rew_max is not None and rew_min is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([iters, iters[::-1]]),
                    y=np.concatenate([rew_max[tick, :steps], rew_min[tick, :steps][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba(33,150,243,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=col_i,
            )

        fig.update_xaxes(title_text="Iteration", row=1, col=col_i)

    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_layout(
        title="MPPI Reward Convergence per Tick",
        **_base_layout(height=380),
    )
    return fig


def plot_reward_across_ticks(
    rew_mean: np.ndarray,
    rew_max: Optional[np.ndarray] = None,
    rew_min: Optional[np.ndarray] = None,
    opt_steps: Optional[np.ndarray] = None,
) -> go.Figure:
    """Plot final-iteration reward trend across all control ticks.

    Args:
        rew_mean: shape (n_ticks, n_iters).
        rew_max:  shape (n_ticks, n_iters), optional.
        rew_min:  shape (n_ticks, n_iters), optional.
        opt_steps: shape (n_ticks,), optional. Used to pick the "final" iteration
                   index per tick. Defaults to last column.

    Returns:
        go.Figure: line chart with tick index on X, final reward on Y.
    """
    n_ticks, n_iters = rew_mean.shape
    if opt_steps is None:
        opt_steps = np.full(n_ticks, n_iters)
    opt_steps = np.asarray(opt_steps).flatten()

    def _final(arr: np.ndarray) -> np.ndarray:
        return np.array([arr[t, max(0, int(opt_steps[t]) - 1)] for t in range(n_ticks)])

    ticks_x = np.arange(n_ticks)
    fig = go.Figure()

    final_mean = _final(rew_mean)
    fig.add_trace(go.Scatter(
        x=ticks_x, y=final_mean,
        mode="lines+markers",
        name="Final rew_mean",
        line=dict(color=BLUE, width=2),
        marker=dict(symbol="square", size=6),
        hovertemplate="Tick %{x}<br>rew_mean=%{y:.5f}<extra></extra>",
    ))

    if rew_max is not None:
        final_max = _final(rew_max)
        fig.add_trace(go.Scatter(
            x=ticks_x, y=final_max,
            mode="lines+markers",
            name="Final rew_max",
            line=dict(color=RED, width=2),
            marker=dict(symbol="circle", size=5),
            hovertemplate="Tick %{x}<br>rew_max=%{y:.5f}<extra></extra>",
        ))

        if rew_min is not None:
            final_min = _final(rew_min)
            # fill between mean and max
            fig.add_trace(go.Scatter(
                x=np.concatenate([ticks_x, ticks_x[::-1]]),
                y=np.concatenate([final_max, final_mean[::-1]]),
                fill="toself",
                fillcolor="rgba(156,39,176,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="Reward Trend Across Control Ticks (final iteration)",
        xaxis=dict(title="Tick Index", dtick=1),
        yaxis=dict(title="Reward"),
        **_base_layout(height=380),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 02 — Cost decomposition (viz/02_tracking_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_decomposition(
    trace_cost: np.ndarray,
    opt_steps: Optional[np.ndarray] = None,
    cost_labels: Optional[list] = None,
) -> go.Figure:
    """Stacked bar chart of cost components per tick (final iteration).

    Args:
        trace_cost: shape (n_ticks, n_iters, n_components) — per-component costs.
        opt_steps: shape (n_ticks,), optional. Selects the final iteration per tick.
        cost_labels: list of n_components label strings. Defaults to ['c0', 'c1', ...].

    Returns:
        go.Figure: stacked bar chart.
    """
    n_ticks, n_iters, n_comp = trace_cost.shape
    if opt_steps is None:
        opt_steps = np.full(n_ticks, n_iters)
    opt_steps = np.asarray(opt_steps).flatten()

    if cost_labels is None:
        cost_labels = [f"c{i}" for i in range(n_comp)]

    # Pick last actual iteration
    tc_final = np.array([trace_cost[t, max(0, int(opt_steps[t]) - 1), :] for t in range(n_ticks)])  # (n_ticks, n_comp)

    ticks_x = list(range(n_ticks))
    comp_colors = [BLUE, ORANGE, GREEN, RED, PURPLE, CYAN,
                   "#795548", "#607D8B", "#E91E63", "#CDDC39"]

    fig = go.Figure()
    for ci in range(n_comp):
        fig.add_trace(go.Bar(
            x=ticks_x,
            y=tc_final[:, ci],
            name=cost_labels[ci],
            marker_color=comp_colors[ci % len(comp_colors)],
            hovertemplate=f"{cost_labels[ci]}: %{{y:.4f}}<br>Tick %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Cost Breakdown per Tick (final MPPI iteration)",
        xaxis=dict(title="Tick Index", dtick=1),
        yaxis=dict(title="Cost"),
        **_base_layout(height=400),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 03 — Trajectory vs reference (viz/03_trajectory_vs_ref.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_timeseries(
    qpos: np.ndarray,
    ref_qpos: Optional[np.ndarray] = None,
    joint_indices: Optional[list] = None,
    time: Optional[np.ndarray] = None,
    tick_boundaries: Optional[np.ndarray] = None,
) -> go.Figure:
    """Plot joint qpos timeseries, optionally overlaid with reference trajectory.

    Args:
        qpos: shape (n_frames, n_dof) — flattened sim trajectory.
              Also accepts (n_ticks, n_substeps, n_dof) — will be reshaped.
        ref_qpos: shape (n_frames, n_dof) — reference trajectory, same length as qpos.
                  Optional.
        joint_indices: list of DOF indices to plot. If None, auto-selects top 8
                       by variance.
        time: shape (n_frames,) — time axis in seconds. If None, uses frame index.
        tick_boundaries: optional array of time values (or frame indices) where
                         vertical dashed lines should be drawn.

    Returns:
        go.Figure: multi-row subplot, one row per joint.
    """
    # Handle 3-D input (n_ticks, n_substeps, n_dof)
    if qpos.ndim == 3:
        n_ticks, n_sub, n_dof = qpos.shape
        if time is not None and time.ndim == 2:
            tick_boundaries = time[:, 0][1:]  # tick start times
            time = time.reshape(-1)
        qpos = qpos.reshape(-1, n_dof)
        if ref_qpos is not None and ref_qpos.ndim == 3:
            ref_qpos = ref_qpos.reshape(-1, ref_qpos.shape[2])

    n_frames, n_dof = qpos.shape
    x_axis = time if time is not None else np.arange(n_frames)
    x_title = "Time (s)" if time is not None else "Frame"

    # Auto-select joints by variance
    if joint_indices is None:
        std_per_dof = np.std(qpos, axis=0)
        active = np.where(std_per_dof > 0.05)[0]
        if len(active) > 8:
            joint_indices = active[np.linspace(0, len(active) - 1, 8, dtype=int)].tolist()
        elif len(active) > 0:
            joint_indices = active.tolist()
        else:
            joint_indices = list(range(min(8, n_dof)))

    n_joints = len(joint_indices)
    fig = make_subplots(rows=n_joints, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)

    for row_i, dof in enumerate(joint_indices, start=1):
        show_legend = row_i == 1
        color = COLORS[row_i % len(COLORS)]

        fig.add_trace(
            go.Scatter(
                x=x_axis, y=qpos[:, dof],
                mode="lines",
                name=f"DOF {dof} sim",
                line=dict(color=color, width=1.2),
                legendgroup=f"dof{dof}",
                showlegend=show_legend,
                hovertemplate=f"DOF {dof}<br>sim=%{{y:.4f}}<br>t=%{{x:.3f}}<extra></extra>",
            ),
            row=row_i, col=1,
        )

        if ref_qpos is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_axis, y=ref_qpos[:, dof],
                    mode="lines",
                    name=f"DOF {dof} ref",
                    line=dict(color="rgba(200,200,200,0.7)", width=1.0, dash="dash"),
                    legendgroup=f"dof{dof}_ref",
                    showlegend=show_legend,
                    hovertemplate=f"DOF {dof}<br>ref=%{{y:.4f}}<br>t=%{{x:.3f}}<extra></extra>",
                ),
                row=row_i, col=1,
            )

            # error fill
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_axis, x_axis[::-1]]),
                    y=np.concatenate([qpos[:, dof], ref_qpos[:, dof][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba(33,150,243,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_i, col=1,
            )

        # Tick boundary lines
        if tick_boundaries is not None:
            for tb in tick_boundaries:
                fig.add_vline(x=float(tb), line=dict(color="rgba(128,128,128,0.4)",
                                                      width=1, dash="dot"),
                              row=row_i, col=1)

        fig.update_yaxes(title_text=f"DOF {dof}", row=row_i, col=1,
                         title_font=dict(size=9))

    fig.update_xaxes(title_text=x_title, row=n_joints, col=1)
    fig.update_layout(
        title="Joint qpos Timeseries: Sim vs Reference",
        height=220 * n_joints,
        **_base_layout(margin=dict(l=70, r=30, t=60, b=50)),
    )
    return fig


def plot_tick_discontinuity_heatmap(
    qpos: np.ndarray,
) -> go.Figure:
    """Heatmap of qpos discontinuities at tick boundaries.

    Args:
        qpos: shape (n_ticks, n_substeps, n_dof) — structured sim trajectory.

    Returns:
        go.Figure: heatmap where X=tick boundary, Y=DOF, Z=|Δqpos|.
    """
    assert qpos.ndim == 3, "qpos must be (n_ticks, n_substeps, n_dof)"
    n_ticks, n_sub, n_dof = qpos.shape
    n_boundaries = n_ticks - 1

    disc = np.zeros((n_dof, n_boundaries))
    for t in range(n_boundaries):
        disc[:, t] = np.abs(qpos[t + 1, 0, :] - qpos[t, -1, :])

    # Filter active DOFs
    active_mask = np.max(disc, axis=1) > 1e-8
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        active_indices = np.arange(n_dof)
    disc_active = disc[active_indices, :]

    x_labels = [f"{t}→{t+1}" for t in range(n_boundaries)]
    y_labels = [f"DOF {d}" for d in active_indices]

    fig = go.Figure(go.Heatmap(
        z=disc_active,
        x=x_labels,
        y=y_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="|Δqpos|"),
        hoverongaps=False,
        hovertemplate="Boundary: %{x}<br>DOF: %{y}<br>|Δqpos|=%{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Tick Boundary Discontinuity: |qpos_start(t+1) − qpos_end(t)|<br>"
              f"<sup>{n_boundaries} boundaries × {len(active_indices)} active DOFs</sup>",
        xaxis=dict(title="Tick Boundary", tickangle=45),
        yaxis=dict(title="DOF"),
        **_base_layout(height=max(300, 20 * len(active_indices) + 120)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 04 — Control analysis (viz/04_control_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ctrl_timeseries(
    ctrl: np.ndarray,
    time: Optional[np.ndarray] = None,
    actuator_indices: Optional[list] = None,
    actuator_labels: Optional[dict] = None,
) -> go.Figure:
    """Plot control signal timeseries for selected actuators.

    Args:
        ctrl: shape (n_ticks, n_substeps, n_actuators) or (n_frames, n_actuators).
        time: shape (n_ticks, n_substeps) or (n_frames,). If None, uses frame index.
        actuator_indices: list of actuator indices to plot. Defaults to 8 spread.
        actuator_labels: dict mapping index → label string. Optional.

    Returns:
        go.Figure: overlaid line traces, one per selected actuator.
    """
    if ctrl.ndim == 3:
        n_ticks, n_sub, n_act = ctrl.shape
        tick_start_times = time[:, 0][1:] if time is not None else None
        ctrl_flat = ctrl.reshape(-1, n_act)
        time_flat = time.reshape(-1) if time is not None else np.arange(n_ticks * n_sub)
    else:
        ctrl_flat = ctrl
        n_act = ctrl.shape[1]
        time_flat = time if time is not None else np.arange(ctrl.shape[0])
        tick_start_times = None

    if actuator_indices is None:
        stride = max(1, n_act // 8)
        actuator_indices = list(range(0, n_act, stride))[:8]

    fig = go.Figure()
    for i, idx in enumerate(actuator_indices):
        label = (actuator_labels or {}).get(idx, f"Act {idx}")
        fig.add_trace(go.Scatter(
            x=time_flat, y=ctrl_flat[:, idx],
            mode="lines",
            name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=1.1),
            hovertemplate=f"{label}<br>ctrl=%{{y:.4f}}<br>t=%{{x:.3f}}<extra></extra>",
        ))

    # Tick boundaries
    if tick_start_times is not None:
        for tb in tick_start_times:
            fig.add_vline(x=float(tb),
                          line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dash"))

    x_title = "Time (s)" if time is not None else "Frame"
    fig.update_layout(
        title="Control Signals — Selected Actuators over Time",
        xaxis=dict(title=x_title),
        yaxis=dict(title="ctrl value"),
        **_base_layout(height=420),
    )
    return fig


def plot_ctrl_effort_vs_optsteps(
    ctrl: np.ndarray,
    opt_steps: np.ndarray,
) -> go.Figure:
    """Per-tick control effort (mean L2 norm) with opt_steps overlay.

    Args:
        ctrl: shape (n_ticks, n_substeps, n_actuators).
        opt_steps: shape (n_ticks,) or (n_ticks, 1).

    Returns:
        go.Figure: grouped bars for effort + rate, with opt_steps line on secondary Y.
    """
    assert ctrl.ndim == 3, "ctrl must be (n_ticks, n_substeps, n_actuators)"
    n_ticks = ctrl.shape[0]
    opt_steps_flat = np.asarray(opt_steps).flatten()

    # Mean L2 norm across actuators, then mean across substeps
    ctrl_l2 = np.linalg.norm(ctrl, axis=2)        # (n_ticks, n_substeps)
    ctrl_effort = ctrl_l2.mean(axis=1)             # (n_ticks,)

    # Mean |Δctrl| rate
    dctrl = np.diff(ctrl, axis=1)                  # (n_ticks, n_substeps-1, n_actuators)
    dctrl_l2 = np.linalg.norm(dctrl, axis=2)       # (n_ticks, n_substeps-1)
    ctrl_rate = dctrl_l2.mean(axis=1)              # (n_ticks,)

    ticks_x = list(range(n_ticks))
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=ticks_x, y=ctrl_effort, name="Mean ‖ctrl‖₂",
               marker_color=BLUE, opacity=0.85,
               hovertemplate="Tick %{x}<br>effort=%{y:.4f}<extra>effort</extra>",
               offsetgroup=0),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=ticks_x, y=ctrl_rate, name="Mean ‖Δctrl‖₂ (rate)",
               marker_color=ORANGE, opacity=0.85,
               hovertemplate="Tick %{x}<br>rate=%{y:.4f}<extra>rate</extra>",
               offsetgroup=1),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ticks_x, y=opt_steps_flat,
                   mode="lines+markers",
                   name="opt_steps",
                   line=dict(color=RED, width=2),
                   marker=dict(size=5),
                   hovertemplate="Tick %{x}<br>opt_steps=%{y}<extra></extra>"),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="group",
        title="Per-Tick Control Effort vs Optimizer Steps",
        xaxis=dict(title="Tick", dtick=1),
        **_base_layout(height=420),
    )
    fig.update_yaxes(title_text="Control magnitude / rate", secondary_y=False)
    fig.update_yaxes(title_text="opt_steps", secondary_y=True)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 07 — Multi-experiment comparison (viz/07_compare_runs.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_compare_reward(
    datasets: dict[str, np.ndarray],
) -> go.Figure:
    """Overlay final-iteration reward per tick across multiple experiments.

    Args:
        datasets: mapping of experiment_label → rew_mean array (n_ticks, n_iters).
                  Optionally include 'rew_min' / 'rew_max' keys as sub-dicts, but
                  accepts simple (n_ticks, n_iters) arrays too — uses last iteration.

    Returns:
        go.Figure: line chart with one trace per experiment.
    """
    fig = go.Figure()

    for i, (label, rew_mean) in enumerate(datasets.items()):
        color = COLORS[i % len(COLORS)]
        rew_final = rew_mean[:, -1]
        ticks_x = list(range(len(rew_final)))

        fig.add_trace(go.Scatter(
            x=ticks_x, y=rew_final,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"{label}<br>reward=%{{y:.5f}}<br>Tick %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        title="Reward per Tick — Multi-Experiment Comparison",
        xaxis=dict(title="Tick Index", dtick=1),
        yaxis=dict(title="Reward (final MPPI iteration)"),
        **_base_layout(height=400),
    )
    return fig


def plot_compare_optsteps(
    datasets: dict[str, np.ndarray],
) -> go.Figure:
    """Grouped bar chart of opt_steps per tick across multiple experiments.

    Args:
        datasets: mapping of experiment_label → opt_steps array (n_ticks,) or (n_ticks, 1).

    Returns:
        go.Figure: grouped bar chart.
    """
    fig = go.Figure()

    labels = list(datasets.keys())
    n_exp = len(labels)

    for i, label in enumerate(labels):
        steps = np.asarray(datasets[label]).flatten()
        ticks_x = list(range(len(steps)))
        color = COLORS[i % len(COLORS)]

        fig.add_trace(go.Bar(
            x=ticks_x,
            y=steps,
            name=label,
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"{label}<br>opt_steps=%{{y}}<br>Tick %{{x}}<extra></extra>",
            offsetgroup=i,
        ))

    fig.update_layout(
        barmode="group",
        title="Optimization Steps per Tick — Multi-Experiment Comparison",
        xaxis=dict(title="Tick Index", dtick=1),
        yaxis=dict(title="opt_steps"),
        **_base_layout(height=400),
    )
    return fig


def plot_compare_cost_breakdown(
    datasets: dict[str, np.ndarray],
    cost_labels: Optional[list] = None,
) -> go.Figure:
    """Side-by-side stacked bar subplots of cost breakdown per experiment.

    Args:
        datasets: mapping of experiment_label → trace_cost array
                  (n_ticks, n_iters, n_components). Uses last iteration.
        cost_labels: list of component label strings. Defaults to ['c0', ...].

    Returns:
        go.Figure: one subplot per experiment, stacked bar by component.
    """
    labels = list(datasets.keys())
    n_exp = len(labels)

    # Infer n_comp from first entry
    first = next(iter(datasets.values()))
    n_comp = first.shape[2] if first.ndim == 3 else first.shape[1]
    if cost_labels is None:
        cost_labels = [f"c{i}" for i in range(n_comp)]

    comp_colors = [BLUE, ORANGE, GREEN, RED, PURPLE, CYAN,
                   "#795548", "#607D8B", "#E91E63", "#CDDC39"]

    fig = make_subplots(
        rows=1, cols=n_exp,
        subplot_titles=labels,
        shared_yaxes=True,
    )

    for col_i, label in enumerate(labels, start=1):
        tc = datasets[label]
        if tc.ndim == 3:
            tc_final = tc[:, -1, :]  # (n_ticks, n_comp)
        else:
            tc_final = tc            # assume already (n_ticks, n_comp)

        n_ticks = tc_final.shape[0]
        ticks_x = list(range(n_ticks))
        show_legend = col_i == 1

        for ci in range(n_comp):
            fig.add_trace(
                go.Bar(
                    x=ticks_x,
                    y=tc_final[:, ci],
                    name=cost_labels[ci],
                    marker_color=comp_colors[ci % len(comp_colors)],
                    legendgroup=cost_labels[ci],
                    showlegend=show_legend,
                    hovertemplate=f"{cost_labels[ci]}: %{{y:.4f}}<br>Tick %{{x}}<extra></extra>",
                    offsetgroup=ci,
                ),
                row=1, col=col_i,
            )
        fig.update_xaxes(title_text="Tick", dtick=1, row=1, col=col_i)

    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_layout(
        barmode="stack",
        title="Cost Breakdown by Experiment (final MPPI iteration)",
        **_base_layout(height=430),
    )
    return fig
