#!/usr/bin/env python3
"""Compare results across SPIDER ablation experiments.

Generates bar charts, line plots, and summary tables for tracking error and reward.

Usage:
    python experiments/compare_results.py \
        --exp_dirs outputs/ablation/p36-tea/baseline outputs/ablation/p36-tea/spider_original outputs/ablation/p36-tea/contact_in_cost \
        --labels baseline spider_original contact_in_cost \
        --output_dir outputs/ablation/comparison
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root so we can import spider modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
FIGSIZE_BAR = (8, 5)
FIGSIZE_LINE = (10, 5)


def load_experiment(exp_dir: str) -> dict | None:
    """Load trajectory npz from an experiment directory."""
    for name in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            data = dict(np.load(path))
            data["_source"] = name
            data["_is_act"] = name.endswith("_act.npz")
            return data
    return None


def compute_tracking_error_over_time(qpos: np.ndarray, embodiment_type: str = "bimanual",
                                      is_act: bool = False) -> dict:
    """Compute per-tick object tracking error (position + orientation).

    qpos shape: (T, ctrl_steps, nq) or (T, nq).
    Returns arrays of shape (T,).
    """
    if qpos.ndim == 3:
        # Use last sub-step of each control tick
        qpos = qpos[:, -1, :]

    # For tracking error we compare against the trajectory's own first frame
    # (drift from initial). This is a simplified metric when no ref is available.
    # In practice the full comparison uses compute_object_tracking_error with a ref.
    # Here we just return the raw object qpos for cross-experiment comparison.

    if embodiment_type == "bimanual":
        if is_act:
            obj_pos = qpos[:, -12:-9]  # right object pos
        else:
            obj_pos = qpos[:, -14:-11]
    else:
        if is_act:
            obj_pos = qpos[:, -6:-3]
        else:
            obj_pos = qpos[:, -7:-4]

    # Displacement from start
    pos_drift = np.linalg.norm(obj_pos - obj_pos[:1], axis=1)
    return {"pos_drift": pos_drift}


def extract_reward_curve(data: dict) -> np.ndarray | None:
    """Extract per-tick reward from trajectory data."""
    for key in ["qpos_rew", "rew_mean"]:
        if key in data:
            arr = data[key]
            if arr.ndim == 1:
                return arr
            elif arr.ndim == 2:
                # (T, iterations) -> take mean across iterations
                return arr.mean(axis=1)
    return None


def extract_improvement_curve(data: dict) -> np.ndarray | None:
    """Extract per-tick improvement from info."""
    if "improvement" in data:
        arr = data["improvement"]
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            return arr.mean(axis=1)
    return None


def plot_bar_final_metrics(experiments: dict, output_dir: str):
    """Bar chart of final metrics across experiments."""
    labels = list(experiments.keys())
    metrics = {}

    for label, data in experiments.items():
        rew_curve = extract_reward_curve(data)
        if rew_curve is not None:
            metrics.setdefault("Final Reward (mean)", []).append(float(rew_curve[-1]))
        else:
            metrics.setdefault("Final Reward (mean)", []).append(0.0)

        # qpos/qvel dist if available
        for key, display in [("qpos_dist", "qpos distance"), ("qvel_dist", "qvel distance")]:
            if key in data:
                arr = data[key]
                val = float(arr[-1].mean()) if arr.ndim > 1 else float(arr[-1])
                metrics.setdefault(display, []).append(val)

    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(labels, values, color=COLORS[:len(labels)], edgecolor="black", linewidth=0.5)
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel(metric_name)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_final_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_reward_over_time(experiments: dict, output_dir: str):
    """Line plot of reward over ticks."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    has_data = False

    for i, (label, data) in enumerate(experiments.items()):
        rew = extract_reward_curve(data)
        if rew is not None:
            ax.plot(rew, label=label, color=COLORS[i % len(COLORS)], linewidth=1.5)
            has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Control Tick")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "reward_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_improvement_over_time(experiments: dict, output_dir: str):
    """Line plot of optimization improvement over ticks."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    has_data = False

    for i, (label, data) in enumerate(experiments.items()):
        imp = extract_improvement_curve(data)
        if imp is not None:
            ax.plot(imp, label=label, color=COLORS[i % len(COLORS)], linewidth=1.5)
            has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Control Tick")
    ax.set_ylabel("Improvement")
    ax.set_title("Optimization Improvement Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "improvement_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_tracking_curves(experiments: dict, output_dir: str):
    """Line plot of qpos/qvel reward (tracking quality) over time."""
    for key, title in [("qpos_rew", "qpos Reward"), ("qvel_rew", "qvel Reward")]:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        has_data = False

        for i, (label, data) in enumerate(experiments.items()):
            if key in data:
                arr = data[key]
                if arr.ndim == 2:
                    curve = arr.mean(axis=1)
                else:
                    curve = arr
                ax.plot(curve, label=label, color=COLORS[i % len(COLORS)], linewidth=1.5)
                has_data = True

        if not has_data:
            plt.close()
            continue

        ax.set_xlabel("Control Tick")
        ax.set_ylabel(title)
        ax.set_title(f"{title} Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{key}_over_time.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def print_summary_table(experiments: dict):
    """Print a text summary table."""
    print(f"\n{'='*80}")
    print(f"{'Experiment':<25} {'Ticks':>6} {'Rew(final)':>12} {'Rew(mean)':>12} {'qpos_rew':>12} {'qvel_rew':>12}")
    print(f"{'-'*80}")

    for label, data in experiments.items():
        rew = extract_reward_curve(data)
        n_ticks = 0

        if "qpos" in data:
            qpos = data["qpos"]
            n_ticks = qpos.shape[0]

        rew_final = f"{float(rew[-1]):.4f}" if rew is not None else "N/A"
        rew_mean = f"{float(rew.mean()):.4f}" if rew is not None else "N/A"

        qpos_rew_str = "N/A"
        if "qpos_rew" in data:
            arr = data["qpos_rew"]
            qpos_rew_str = f"{float(arr.mean()):.4f}"

        qvel_rew_str = "N/A"
        if "qvel_rew" in data:
            arr = data["qvel_rew"]
            qvel_rew_str = f"{float(arr.mean()):.4f}"

        print(f"{label:<25} {n_ticks:>6} {rew_final:>12} {rew_mean:>12} {qpos_rew_str:>12} {qvel_rew_str:>12}")

    print(f"{'='*80}\n")


def save_summary_csv(experiments: dict, output_dir: str):
    """Save summary stats as CSV."""
    rows = []
    for label, data in experiments.items():
        row = {"experiment": label}
        if "qpos" in data:
            row["n_ticks"] = data["qpos"].shape[0]

        rew = extract_reward_curve(data)
        if rew is not None:
            row["rew_final"] = float(rew[-1])
            row["rew_mean"] = float(rew.mean())
            row["rew_std"] = float(rew.std())

        for key in ["qpos_rew", "qvel_rew", "qpos_dist", "qvel_dist"]:
            if key in data:
                arr = data[key]
                row[f"{key}_mean"] = float(arr.mean())
                row[f"{key}_std"] = float(arr.std())
                final = arr[-1].mean() if arr.ndim > 1 else arr[-1]
                row[f"{key}_final"] = float(final)

        rows.append(row)

    # Write CSV manually (no pandas dependency)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)

    path = os.path.join(output_dir, "summary.csv")
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SPIDER ablation experiments")
    parser.add_argument("--exp_dirs", nargs="+", required=True,
                        help="Experiment output directories")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each experiment (defaults to dir basename)")
    parser.add_argument("--output_dir", default="outputs/ablation/comparison",
                        help="Where to save comparison plots")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.exp_dirs):
        print("Error: --labels count must match --exp_dirs count")
        sys.exit(1)

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.exp_dirs]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all experiments
    experiments = {}
    for label, exp_dir in zip(labels, args.exp_dirs):
        data = load_experiment(exp_dir)
        if data is None:
            print(f"⚠ Skipping '{label}': no trajectory file in {exp_dir}")
            continue
        experiments[label] = data
        print(f"✓ Loaded '{label}' from {exp_dir} ({data['_source']})")

    if len(experiments) < 2:
        print(f"\nNeed at least 2 experiments to compare, got {len(experiments)}.")
        if len(experiments) == 1:
            print("Generating single-experiment summary anyway.\n")
        else:
            sys.exit(1)

    # Summary table
    print_summary_table(experiments)

    # Plots
    print("Generating plots...")
    plot_bar_final_metrics(experiments, args.output_dir)
    plot_reward_over_time(experiments, args.output_dir)
    plot_improvement_over_time(experiments, args.output_dir)
    plot_tracking_curves(experiments, args.output_dir)

    # CSV
    save_summary_csv(experiments, args.output_dir)

    print(f"\n✅ All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
