"""Sweep noise scale multipliers for pure MPPI to see convergence vs covariance size."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_closed_loop_compare import make_config
from examples.run_mjwp import main as run_mjwp_main

TASK = "P0001_4bf4e21a-obj96945373046044"
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
OUTPUT_DIR = "outputs/noise_scale_sweep"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for scale in SCALES:
        label = f"mppi_scale_{scale:.2f}"
        print(f"\n=== {label} ===")
        
        run_dir = os.path.join(OUTPUT_DIR, label, "seed_0")
        os.makedirs(run_dir, exist_ok=True)

        ctrl_steps = int(round(0.4 / 0.01))
        max_sim_steps = 1 * ctrl_steps  # 1 tick

        config = make_config(
            task=TASK,
            dataset_name="hot3d",
            seed=0,
            optimizer_type="mppi",
            output_dir=run_dir,
            num_samples=1024,
            max_iters=32,
            final_noise_scale=1.0,  # pure MPPI, no annealing
            max_sim_steps=max_sim_steps,
        )
        # Override noise scales
        config.first_ctrl_noise_scale *= scale
        config.last_ctrl_noise_scale *= scale

        t0 = time.perf_counter()
        run_mjwp_main(config)
        dt = time.perf_counter() - t0
        print(f"    OK ({dt:.1f}s)")

    # Print results
    print("\n\n=== RESULTS ===")
    print(f"{'Scale':>8s} {'iter0':>8s} {'iter8':>8s} {'iter16':>8s} {'iter31':>8s} {'Δ':>8s}")
    print("-" * 55)
    for scale in SCALES:
        label = f"mppi_scale_{scale:.2f}"
        path = os.path.join(OUTPUT_DIR, label, "seed_0", "trajectory_mjwp.npz")
        if os.path.exists(path):
            d = np.load(path)
            rew = d['rew_u0'][0]
            n = int(d['opt_steps'][0])
            delta = rew[min(n-1,31)] - rew[0]
            print(f"{scale:8.2f} {rew[0]:8.4f} {rew[min(8,n-1)]:8.4f} {rew[min(16,n-1)]:8.4f} {rew[min(n-1,31)]:8.4f} {delta:+8.4f}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SCALES)))

    for i, scale in enumerate(SCALES):
        label = f"mppi_scale_{scale:.2f}"
        path = os.path.join(OUTPUT_DIR, label, "seed_0", "trajectory_mjwp.npz")
        if os.path.exists(path):
            d = np.load(path)
            rew = d['rew_u0'][0]
            n = int(d['opt_steps'][0])
            ax.plot(range(n), rew[:n], color=colors[i], label=f'scale={scale}', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('rew_u0 (exploit)', fontsize=12)
    ax.set_title('Pure MPPI: Noise Scale Sweep (Tick 0, 1024 samples)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_sweep.png'), dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {OUTPUT_DIR}/convergence_sweep.png")

if __name__ == "__main__":
    main()
