"""
ui/metric_cards.py — SPIDER Dashboard Metric Cards
===================================================
Renders the four key experiment metrics using Streamlit's st.metric().

Expected `metrics` dict schema
-------------------------------
{
    "opt_steps": {
        "mean": float,          # mean optimization steps across ticks
        "std":  float,          # std of opt_steps
        "delta": float | None,  # comparison to previous/baseline (optional)
    },
    "reward": {
        "mean": float,
        "delta": float | None,
    },
    "discontinuity": {
        "mean": float,          # mean |Δqpos| at tick boundaries (lower is better)
        "delta": float | None,
    },
    "variance": {
        "value": float,         # overall qpos/reward variance (lower is better)
        "delta": float | None,
    },
}

Usage
-----
    from ui.metric_cards import render_metric_cards
    render_metric_cards(metrics)
"""

from __future__ import annotations

import streamlit as st


def render_metric_cards(metrics: dict) -> None:
    """Render the four SPIDER key-metric cards in a 4-column layout.

    Args:
        metrics: dict matching the schema described in the module docstring.
                 Missing keys are shown as "N/A". Optional sub-keys (delta,
                 std) are silently omitted if absent.

    Returns:
        None. Side-effect: renders Streamlit metric widgets.
    """
    col1, col2, col3, col4 = st.columns(4)

    # ── 1. Opt Steps ───────────────────────────────────────────────────
    opt = metrics.get("opt_steps", {})
    opt_mean = opt.get("mean")
    opt_std = opt.get("std")
    opt_delta = opt.get("delta")

    if opt_mean is not None:
        label_val = f"{opt_mean:.1f}"
        if opt_std is not None:
            label_val += f" ± {opt_std:.1f}"
        delta_str = f"{opt_delta:+.1f}" if opt_delta is not None else None
    else:
        label_val = "N/A"
        delta_str = None

    with col1:
        st.metric(
            label="⚙️ Opt Steps",
            value=label_val,
            delta=delta_str,
            help="Mean (± std) optimization steps per tick across all ticks. "
                 "Higher = optimizer worked harder.",
        )

    # ── 2. Reward ──────────────────────────────────────────────────────
    rew = metrics.get("reward", {})
    rew_mean = rew.get("mean")
    rew_delta = rew.get("delta")

    if rew_mean is not None:
        rew_val = f"{rew_mean:.4f}"
        rew_delta_str = f"{rew_delta:+.4f}" if rew_delta is not None else None
    else:
        rew_val = "N/A"
        rew_delta_str = None

    with col2:
        st.metric(
            label="🏆 Reward",
            value=rew_val,
            delta=rew_delta_str,
            help="Mean final-iteration reward averaged across all ticks. "
                 "Higher is better.",
        )

    # ── 3. Discontinuity ───────────────────────────────────────────────
    disc = metrics.get("discontinuity", {})
    disc_mean = disc.get("mean")
    disc_delta = disc.get("delta")

    if disc_mean is not None:
        disc_val = f"{disc_mean:.5f}"
        # Invert delta_color: lower discontinuity is better
        # Streamlit colors delta green if positive; we want negative to be green here.
        if disc_delta is not None:
            disc_delta_str = f"{disc_delta:+.5f}"
            # Trick: negate so green = improvement (reduction)
            disc_delta_display = f"{-disc_delta:+.5f}"
        else:
            disc_delta_str = None
            disc_delta_display = None
    else:
        disc_val = "N/A"
        disc_delta_display = None

    with col3:
        st.metric(
            label="📐 Discontinuity",
            value=disc_val,
            delta=disc_delta_display,
            delta_color="normal",   # green = improvement (negated above)
            help="Mean |Δqpos| at tick boundaries. Lower is better — "
                 "smooth transitions between control ticks.",
        )

    # ── 4. Variance ────────────────────────────────────────────────────
    var = metrics.get("variance", {})
    var_val = var.get("value")
    var_delta = var.get("delta")

    if var_val is not None:
        var_str = f"{var_val:.5f}"
        if var_delta is not None:
            var_delta_display = f"{-var_delta:+.5f}"  # negate: lower variance is better
        else:
            var_delta_display = None
    else:
        var_str = "N/A"
        var_delta_display = None

    with col4:
        st.metric(
            label="📊 Variance",
            value=var_str,
            delta=var_delta_display,
            delta_color="normal",
            help="Overall trajectory variance (qpos std across ticks). "
                 "Lower is better — stable, consistent rollouts.",
        )
