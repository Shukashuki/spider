#!/usr/bin/env python3
"""Compare multi-task ablation results with focus on convergence speed.

Generates:
  - Per-task reward convergence curves (tick-level)
  - Per-task intra-tick optimization curves (iteration-level, first & last tick)
  - Cross-task summary bar chart (final reward + wall-clock time)
  - Summary CSV

Usage:
    uv run python experiments/compare_multi_task.py --base_dir outputs/multi_task
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["spider_original", "baseline", "cma_with_ref"]
METHOD_LABELS = {
    "spider_original": "MPPI + Ref",
    "baseline": "MPPI + Zero",
    "cma_with_ref": "CMA + Ref",
}
COLORS = {
    "spider_original": "#4C72B0",
    "baseline": "#DD8452",
    "cma_with_ref": "#55A868",
}
FIGSIZE = (10, 5)


def load_run(run_dir: str) -> dict | None:
    for name in ["trajectory_mjwp_act.npz", "trajectory_mjwp.npz"]:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return dict(np.load(path))
    return None


def parse_wall_time(run_dir: str) -> float | None:
    """Extract total wall-clock time from run.log."""
    log_path = os.path.join(run_dir, "run.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            if "Total time:" in line:
                try:
                    return float(line.split("Total time:")[1].strip().rstrip("s"))
                except ValueError:
                    pass
    return None


def get_tick_rewards(data: dict) -> np.ndarray:
    """Get per-tick reward at the actual last optimization iteration."""
    rew = data["rew_mean"]  # (ticks, max_iters)
    opt = data["opt_steps"]
    if opt.ndim == 2:
        steps = opt[:, 0].astype(int)
    else:
        steps = opt.astype(int)

    result = []
    for t in range(rew.shape[0]):
        n = int(steps[t])
        if 0 < n <= rew.shape[1]:
            result.append(rew[t, n - 1])
        else:
            row = rew[t]
            valid = row[row != 0]
            result.append(valid[-1] if len(valid) > 0 else 0.0)
    return np.array(result)


def get_iter_curve(data: dict, tick: int) -> np.ndarray:
    """Get the optimization iteration curve for a specific tick."""
    rew = data["rew_mean"]  # (ticks, max_iters)
    opt = data["opt_steps"]
    if opt.ndim == 2:
        n = int(opt[tick, 0])
    else:
        n = int(opt[tick])
    row = rew[tick, :n]
    # filter NaN
    return row[~np.isnan(row)]


def get_plan_times(run_dir: str) -> list[float]:
    """Extract per-tick plan times from run.log."""
    log_path = os.path.join(run_dir, "run.log")
    times = []
    if not os.path.exists(log_path):
        return times
    with open(log_path) as f:
        for line in f:
            if "plan time:" in line:
                # may have multiple entries on one line (carriage return)
                parts = line.split("plan time:")
                for part in parts[1:]:
                    try:
                        val = float(part.split("s")[0].strip())
                        times.append(val)
                    except ValueError:
                        pass
    return times


def plot_tick_convergence(task: str, runs: dict, output_dir: str):
    """Reward over control ticks for one task."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method in METHODS:
        if method not in runs:
            continue
        curve = get_tick_rewards(runs[method]["data"])
        label = METHOD_LABELS.get(method, method)
        ax.plot(curve, label=label, color=COLORS[method], linewidth=2)

    ax.set_xlabel("Control Tick", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(f"Reward Convergence — {task}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{task}_tick_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_iter_convergence(task: str, runs: dict, output_dir: str):
    """Intra-tick optimization curves (first tick and last tick)."""
    n_ticks = None
    for method in METHODS:
        if method in runs:
            n_ticks = runs[method]["data"]["rew_mean"].shape[0]
            break
    if n_ticks is None:
        return

    for tick_idx, tick_label in [(0, "first"), (n_ticks - 1, "last")]:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method in METHODS:
            if method not in runs:
                continue
            curve = get_iter_curve(runs[method]["data"], tick_idx)
            if len(curve) == 0:
                continue
            label = METHOD_LABELS.get(method, method)
            ax.plot(curve, label=f"{label} ({len(curve)} iters)",
                    color=COLORS[method], linewidth=2)

        ax.set_xlabel("Optimization Iteration", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title(f"Intra-Tick Optimization — {task} (tick {tick_idx}, {tick_label})",
                      fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{task}_iter_{tick_label}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  {path}")


def plot_plan_time(task: str, runs: dict, output_dir: str):
    """Per-tick planning time comparison."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method in METHODS:
        if method not in runs:
            continue
        times = runs[method]["plan_times"]
        if not times:
            continue
        label = METHOD_LABELS.get(method, method)
        ax.plot(times, label=f"{label} (avg {np.mean(times):.1f}s)",
                color=COLORS[method], linewidth=2)

    ax.set_xlabel("Control Tick", fontsize=12)
    ax.set_ylabel("Plan Time (s)", fontsize=12)
    ax.set_title(f"Per-Tick Planning Time — {task}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{task}_plan_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_cross_task_summary(all_runs: dict, output_dir: str):
    """Bar chart: final reward and total time across all tasks."""
    tasks = sorted(all_runs.keys())
    n_tasks = len(tasks)
    n_methods = len(METHODS)
    x = np.arange(n_tasks)
    width = 0.25

    # --- Final reward ---
    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 3), 5))
    for i, method in enumerate(METHODS):
        vals = []
        for task in tasks:
            if method in all_runs[task]:
                curve = get_tick_rewards(all_runs[task][method]["data"])
                vals.append(curve[-1])
            else:
                vals.append(0)
        label = METHOD_LABELS.get(method, method)
        bars = ax.bar(x + i * width, vals, width, label=label,
                       color=COLORS[method], edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Final Reward", fontsize=12)
    ax.set_title("Final Reward Across Tasks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "cross_task_final_reward.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")

    # --- Wall-clock time ---
    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 3), 5))
    for i, method in enumerate(METHODS):
        vals = []
        for task in tasks:
            if method in all_runs[task]:
                wt = all_runs[task][method]["wall_time"]
                vals.append(wt if wt else 0)
            else:
                vals.append(0)
        label = METHOD_LABELS.get(method, method)
        bars = ax.bar(x + i * width, vals, width, label=label,
                       color=COLORS[method], edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Wall-Clock Time (s)", fontsize=12)
    ax.set_title("Total Optimization Time Across Tasks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "cross_task_wall_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def save_summary_csv(all_runs: dict, output_dir: str):
    tasks = sorted(all_runs.keys())
    path = os.path.join(output_dir, "summary.csv")
    with open(path, "w") as f:
        f.write("task,method,rew_final,rew_mean,wall_time_s,avg_opt_steps,avg_plan_time_s\n")
        for task in tasks:
            for method in METHODS:
                if method not in all_runs[task]:
                    continue
                r = all_runs[task][method]
                curve = get_tick_rewards(r["data"])
                opt = r["data"]["opt_steps"]
                if opt.ndim == 2:
                    avg_opt = float(opt[:, 0].mean())
                else:
                    avg_opt = float(opt.mean())
                pt = r["plan_times"]
                avg_pt = float(np.mean(pt)) if pt else 0
                wt = r["wall_time"] or 0
                f.write(f"{task},{method},{curve[-1]:.4f},{np.nanmean(curve):.4f},"
                        f"{wt:.1f},{avg_opt:.1f},{avg_pt:.1f}\n")
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="outputs/multi_task")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "comparison")
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover tasks
    tasks = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d)) and d != "comparison"
    ])
    print(f"Tasks found: {tasks}\n")

    all_runs: dict[str, dict] = {}
    for task in tasks:
        all_runs[task] = {}
        for method in METHODS:
            run_dir = os.path.join(args.base_dir, task, method)
            data = load_run(run_dir)
            if data is None:
                print(f"⚠ Missing: {task}/{method}")
                continue
            all_runs[task][method] = {
                "data": data,
                "wall_time": parse_wall_time(run_dir),
                "plan_times": get_plan_times(run_dir),
            }
            wt = all_runs[task][method]["wall_time"]
            print(f"✓ {task}/{method} — wall={wt:.0f}s" if wt else f"✓ {task}/{method}")

    print("\nGenerating plots...")

    for task in tasks:
        if not all_runs[task]:
            continue
        plot_tick_convergence(task, all_runs[task], args.output_dir)
        plot_iter_convergence(task, all_runs[task], args.output_dir)
        plot_plan_time(task, all_runs[task], args.output_dir)

    if len(tasks) > 1:
        plot_cross_task_summary(all_runs, args.output_dir)

    save_summary_csv(all_runs, args.output_dir)

    # Print summary table
    print(f"\n{'='*85}")
    print(f"{'Task':<18} {'Method':<18} {'Rew(final)':>12} {'Rew(mean)':>12} {'Wall(s)':>10} {'AvgOpt':>8}")
    print(f"{'-'*85}")
    for task in tasks:
        for method in METHODS:
            if method not in all_runs[task]:
                continue
            r = all_runs[task][method]
            curve = get_tick_rewards(r["data"])
            opt = r["data"]["opt_steps"]
            avg_opt = float(opt[:, 0].mean()) if opt.ndim == 2 else float(opt.mean())
            wt = r["wall_time"] or 0
            label = METHOD_LABELS.get(method, method)
            print(f"{task:<18} {label:<18} {curve[-1]:>12.4f} {np.nanmean(curve):>12.4f} {wt:>10.0f} {avg_opt:>8.1f}")
        print()
    print(f"{'='*85}")
    print(f"\n✅ All outputs: {args.output_dir}/")


if __name__ == "__main__":
    main()
