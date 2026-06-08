"""LQR Benchmark v3: Reproduce Zeng et al. experimental setup.

Paper: "The Speed-Optimality Trade-Off of MPPI: A Case Study in LQR"
       Xiong Zeng, Necmiye Ozay, Mario Sznaier (UMich / Northeastern)

Key differences from our v2:
  - Optimize in GAIN SPACE: K ∈ R^{m×n}, not trajectory u ∈ R^{H×m}
  - Cost = tr(P_K) (infinite-horizon Lyapunov cost), not finite-horizon rollout
  - Uniform sampling on sphere ‖ΔK‖_F = r (not Gaussian)
  - N=96 samples, λ=0.01, 100 iterations, 10 trials
  - MPPI-CMA: η_μ=1.0, η_Σ=0.3, init Σ=0.5*I

Experiments:
  Fig 1: Fix λ=0.01, sweep σ ∈ {0.125, 0.25, 0.5, 1.0}
  Fig 2: Fixed σ vs annealed (0.5→0.05)
  Fig 3: MPPI vs annealed vs MPPI-CMA
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


# ============================================================
# LQR Problem (gain-space optimization)
# ============================================================

class LQRGainProblem:
    """Infinite-horizon LQR: optimize K directly, cost = tr(P_K)."""

    def __init__(self, n=4, m=2, seed=42):
        rng = np.random.RandomState(seed)
        # Random stable-ish system (paper doesn't specify, use double integrator)
        # Double integrator: state = [x, y, vx, vy]
        dt = 0.1
        self.n, self.m = n, m
        self.A = np.eye(n)
        self.A[0, 2] = dt; self.A[1, 3] = dt
        self.B = np.zeros((n, m))
        self.B[2, 0] = dt; self.B[3, 1] = dt

        self.Q = np.diag([10., 10., 1., 1.])
        self.R = np.eye(m) * 0.1

        # Optimal gain
        self.P_star = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K_star = np.linalg.inv(self.R + self.B.T @ self.P_star @ self.B) @ (self.B.T @ self.P_star @ self.A)
        self.J_star = np.trace(self.P_star)

        # Initial stabilizing gain (LQR optimal + small perturbation)
        # Start far enough for gap ~10¹ (paper's Fig 1 range)
        self.K0 = self.K_star + rng.randn(m, n) * 1.5

        print(f"System: n={n}, m={m}, d={n*m}")
        print(f"J* = tr(P*) = {self.J_star:.4f}")
        print(f"J(K0) = {self.cost(self.K0):.4f}")
        print(f"ρ(A-BK*) = {max(abs(np.linalg.eigvals(self.A - self.B @ self.K_star))):.4f}")
        print(f"ρ(A-BK0) = {max(abs(np.linalg.eigvals(self.A - self.B @ self.K0))):.4f}")

    def cost(self, K):
        """tr(P_K) where A_K^T P_K A_K - P_K + Q + K^T R K = 0."""
        AK = self.A - self.B @ K
        eigs = np.abs(np.linalg.eigvals(AK))
        if np.max(eigs) >= 1.0:
            return float('inf')
        QK = self.Q + K.T @ self.R @ K
        try:
            PK = solve_discrete_lyapunov(AK.T, QK)
            return np.trace(PK)
        except np.linalg.LinAlgError:
            return float('inf')

    def cost_batch(self, K_batch_flat, K_shape):
        """Batch cost evaluation. K_batch_flat: (N, m*n) -> costs: (N,)."""
        N = K_batch_flat.shape[0]
        costs = np.zeros(N)
        for i in range(N):
            K = K_batch_flat[i].reshape(K_shape)
            costs[i] = self.cost(K)
        return costs


# ============================================================
# Optimizers (gain-space, following paper's setup)
# ============================================================

def sample_sphere(N, d, r, rng):
    """Sample uniformly on sphere ‖ΔK‖_F = r."""
    z = rng.randn(N, d)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z * r


def run_mppi_fixed_sigma(prob, sigma, lam=0.01, N=96, iters=100, seed=0):
    """Pure MPPI with fixed sigma (uniform sphere sampling)."""
    rng = np.random.RandomState(seed)
    m, n = prob.m, prob.n
    d = m * n
    K = prob.K0.copy()
    gaps = []

    for it in range(iters):
        J_K = prob.cost(K)
        gaps.append(J_K - prob.J_star)

        # Sample perturbations on sphere
        DK = sample_sphere(N, d, sigma, rng)
        K_flat = K.flatten()

        # Evaluate
        candidates = K_flat[None, :] + DK  # (N, d)
        costs = prob.cost_batch(candidates, (m, n))

        # Handle inf costs
        valid = np.isfinite(costs)
        if valid.sum() == 0:
            continue

        # MPPI weights
        costs_valid = costs.copy()
        costs_valid[~valid] = costs_valid[valid].max() + 1e6
        J_min = costs_valid[valid].min()
        w = np.exp(-(costs_valid - J_min) / lam)
        w[~valid] = 0.0
        w_sum = w.sum()
        if w_sum < 1e-30:
            continue
        w /= w_sum

        # Update mean
        K_flat = K_flat + (w[:, None] * DK).sum(axis=0)
        K = K_flat.reshape(m, n)

    # Final
    gaps.append(prob.cost(K) - prob.J_star)
    return gaps


def run_mppi_annealed(prob, sigma_start, sigma_end, lam=0.01, N=96, iters=100, seed=0):
    """MPPI with linear annealing sigma_start -> sigma_end."""
    rng = np.random.RandomState(seed)
    m, n = prob.m, prob.n
    d = m * n
    K = prob.K0.copy()
    gaps = []

    for it in range(iters):
        J_K = prob.cost(K)
        gaps.append(J_K - prob.J_star)

        # Linear annealing
        sigma = sigma_start + (sigma_end - sigma_start) * (it / max(iters - 1, 1))

        DK = sample_sphere(N, d, sigma, rng)
        K_flat = K.flatten()
        candidates = K_flat[None, :] + DK
        costs = prob.cost_batch(candidates, (m, n))

        valid = np.isfinite(costs)
        if valid.sum() == 0:
            continue
        costs_valid = costs.copy()
        costs_valid[~valid] = costs_valid[valid].max() + 1e6
        J_min = costs_valid[valid].min()
        w = np.exp(-(costs_valid - J_min) / lam)
        w[~valid] = 0.0
        w_sum = w.sum()
        if w_sum < 1e-30:
            continue
        w /= w_sum

        K_flat = K_flat + (w[:, None] * DK).sum(axis=0)
        K = K_flat.reshape(m, n)

    gaps.append(prob.cost(K) - prob.J_star)
    return gaps


def run_mppi_cma(prob, sigma0=0.5, lam=0.01, N=96, iters=100, seed=0,
                 eta_mu=1.0, eta_sigma=0.3):
    """MPPI-CMA: Gaussian sampling with covariance adaptation."""
    rng = np.random.RandomState(seed)
    m, n = prob.m, prob.n
    d = m * n
    K = prob.K0.copy()
    mu = K.flatten().copy()
    Sigma = (sigma0 ** 2) * np.eye(d)
    jitter = 1e-6
    gaps = []

    for it in range(iters):
        J_K = prob.cost(mu.reshape(m, n))
        gaps.append(J_K - prob.J_star)

        # Sample from N(0, Sigma)
        try:
            L = np.linalg.cholesky(Sigma + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.maximum(np.diag(Sigma), jitter)))

        z = rng.randn(N, d)
        eps = z @ L.T  # (N, d)

        candidates = mu[None, :] + eps  # (N, d)
        costs = prob.cost_batch(candidates, (m, n))

        valid = np.isfinite(costs)
        if valid.sum() == 0:
            continue
        costs_c = costs.copy()
        costs_c[~valid] = costs_c[valid].max() + 1e6
        J_min = costs_c[valid].min()
        w = np.exp(-(costs_c - J_min) / lam)
        w[~valid] = 0.0
        w_sum = w.sum()
        if w_sum < 1e-30:
            continue
        w /= w_sum

        # Covariance update (before mean, as paper specifies)
        weighted_eps = eps * w[:, None]  # (N, d)
        Sigma_sample = (eps * np.sqrt(w)[:, None]).T @ (eps * np.sqrt(w)[:, None])
        Sigma = (1 - eta_sigma) * Sigma + eta_sigma * Sigma_sample + jitter * np.eye(d)
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Mean update
        mu_new = (w[:, None] * candidates).sum(axis=0)
        mu = (1 - eta_mu) * mu + eta_mu * mu_new

    gaps.append(prob.cost(mu.reshape(m, n)) - prob.J_star)
    return gaps


# ============================================================
# Main: Reproduce Fig 1, 2, 3
# ============================================================

def main():
    print("=" * 60)
    print("Reproducing Zeng et al. LQR Gain-Space Benchmark")
    print("=" * 60)

    prob = LQRGainProblem(n=4, m=2)

    N = 96
    lam = 0.01
    iters = 100
    trials = 10

    # ---- Fig 1: Sweep sigma, fixed MPPI ----
    print("\n--- Fig 1: MPPI sensitivity, sweep σ ---")
    sigmas_fig1 = [0.125, 0.25, 0.5, 1.0]
    colors_fig1 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig1_data = {}
    for sig in sigmas_fig1:
        runs = []
        for s in range(trials):
            g = run_mppi_fixed_sigma(prob, sigma=sig, lam=lam, N=N, iters=iters, seed=s)
            runs.append(g)
        fig1_data[sig] = np.array(runs)
        fm = fig1_data[sig][:, -1].mean()
        print(f"  σ={sig:5.3f}: final gap = {fm:.6f}")

    # ---- Fig 2: Fixed sigma vs annealed ----
    print("\n--- Fig 2: Fixed σ vs annealed ---")
    fig2_configs = {
        "fixed σ=0.05": lambda s: run_mppi_fixed_sigma(prob, 0.05, lam, N, iters, s),
        "fixed σ=0.5": lambda s: run_mppi_fixed_sigma(prob, 0.5, lam, N, iters, s),
        "fixed σ=1.0": lambda s: run_mppi_fixed_sigma(prob, 1.0, lam, N, iters, s),
        "annealed 0.5→0.05": lambda s: run_mppi_annealed(prob, 0.5, 0.05, lam, N, iters, s),
    }
    colors_fig2 = {"fixed σ=0.05": "#1f77b4", "fixed σ=0.5": "#2ca02c",
                   "fixed σ=1.0": "#d62728", "annealed 0.5→0.05": "#9467bd"}

    fig2_data = {}
    for name, fn in fig2_configs.items():
        runs = [fn(s) for s in range(trials)]
        fig2_data[name] = np.array(runs)
        fm = fig2_data[name][:, -1].mean()
        print(f"  {name:25s}: final gap = {fm:.6f}")

    # ---- Fig 3: MPPI vs annealed vs MPPI-CMA ----
    print("\n--- Fig 3: MPPI vs annealed vs MPPI-CMA ---")
    fig3_configs = {
        "MPPI fixed σ=0.05": lambda s: run_mppi_fixed_sigma(prob, 0.05, lam, N, iters, s),
        "MPPI fixed σ=0.5": lambda s: run_mppi_fixed_sigma(prob, 0.5, lam, N, iters, s),
        "MPPI fixed σ=1.0": lambda s: run_mppi_fixed_sigma(prob, 1.0, lam, N, iters, s),
        "MPPI annealed 0.5→0.05": lambda s: run_mppi_annealed(prob, 0.5, 0.05, lam, N, iters, s),
        "MPPI-CMA (η_μ=1.0, η_Σ=0.3)": lambda s: run_mppi_cma(
            prob, sigma0=0.5, lam=lam, N=N, iters=iters, seed=s, eta_mu=1.0, eta_sigma=0.3),
    }
    colors_fig3 = {
        "MPPI fixed σ=0.05": "#1f77b4",
        "MPPI fixed σ=0.5": "#2ca02c",
        "MPPI fixed σ=1.0": "#d62728",
        "MPPI annealed 0.5→0.05": "#9467bd",
        "MPPI-CMA (η_μ=1.0, η_Σ=0.3)": "#e377c2",
    }

    fig3_data = {}
    for name, fn in fig3_configs.items():
        runs = [fn(s) for s in range(trials)]
        fig3_data[name] = np.array(runs)
        fm = fig3_data[name][:, -1].mean()
        print(f"  {name:35s}: final gap = {fm:.6f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Fig 1
    ax = axes[0]
    for sig, col in zip(sigmas_fig1, colors_fig1):
        data = fig1_data[sig]
        itr = np.arange(data.shape[1])
        m = np.maximum(data.mean(axis=0), 1e-8)
        s = data.std(axis=0)
        ax.semilogy(itr, m, label=f"σ={sig}", color=col, linewidth=2)
        ax.fill_between(itr, np.maximum(m - s, 1e-8), m + s, alpha=0.15, color=col)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost gap J(K) - J* (log)")
    ax.set_title(f"Fig 1: MPPI sensitivity (λ={lam}, N={N})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Fig 2
    ax = axes[1]
    for name, data in fig2_data.items():
        itr = np.arange(data.shape[1])
        m = np.maximum(data.mean(axis=0), 1e-8)
        s = data.std(axis=0)
        ax.semilogy(itr, m, label=name, color=colors_fig2[name], linewidth=2)
        ax.fill_between(itr, np.maximum(m - s, 1e-8), m + s, alpha=0.15, color=colors_fig2[name])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost gap J(K) - J* (log)")
    ax.set_title(f"Fig 2: Fixed σ vs annealed (λ={lam}, N={N})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Fig 3
    ax = axes[2]
    for name, data in fig3_data.items():
        itr = np.arange(data.shape[1])
        m = np.maximum(data.mean(axis=0), 1e-8)
        s = data.std(axis=0)
        ax.semilogy(itr, m, label=name, color=colors_fig3[name], linewidth=2)
        ax.fill_between(itr, np.maximum(m - s, 1e-8), m + s, alpha=0.15, color=colors_fig3[name])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost gap J(K) - J* (log)")
    ax.set_title(f"Fig 3: MPPI vs annealed vs MPPI-CMA (λ={lam}, N={N})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Reproducing Zeng et al. — LQR Gain-Space Optimization\n"
        f"N={N}, λ={lam}, {iters} iters, {trials} trials, 4D double integrator",
        fontsize=12)
    plt.tight_layout()
    out = "/home/roy/.openclaw/workspace/spider/outputs/lqr_benchmark_v3_zeng.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
