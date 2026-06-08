"""LQR Benchmark: Pure MPPI vs DIAL-MPC vs CMA-ES (rank) vs MPPI-CMA.

Tests optimizer correctness on a discrete-time LQR problem with known
optimal solution. No SPIDER dependencies — pure numpy/scipy/torch.

System: x_{k+1} = A x_k + B u_k + w_k  (w_k = process noise)
Cost:   J = sum_{k=0}^{H-1} (x_k^T Q x_k + u_k^T R u_k) + x_H^T Q x_H

Includes robustness tests:
  - Nominal (no noise)
  - Process noise (stochastic dynamics)
  - Model mismatch (optimizer uses wrong A/B)
  - Non-quadratic cost landscape (added local minima)
"""

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are


# ============================================================
# LQR Problem Setup
# ============================================================

@dataclass
class LQRProblem:
    """Discrete-time LQR with known optimal cost."""
    nx: int = 4
    nu: int = 2
    horizon: int = 20
    dt: float = 0.1
    # Robustness settings
    process_noise_std: float = 0.0       # additive Gaussian noise on dynamics
    model_mismatch: float = 0.0          # fractional error in A/B used by optimizer
    nonquadratic: bool = False           # add sinusoidal bumps to cost landscape

    def __post_init__(self):
        # Double integrator (2D): state = [x, y, vx, vy]
        self.A = np.eye(self.nx)
        self.A[0, 2] = self.dt
        self.A[1, 3] = self.dt
        self.B = np.zeros((self.nx, self.nu))
        self.B[2, 0] = self.dt
        self.B[3, 1] = self.dt
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])
        self.R = np.eye(self.nu) * 0.1
        # Solve DARE for optimal gain (nominal)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ (self.B.T @ self.P @ self.A)
        # Initial state
        self.x0 = np.array([1.0, -0.5, 0.2, -0.1])

    def optimal_cost(self) -> float:
        """Compute optimal cost via forward simulation with K (nominal, no noise)."""
        x = self.x0.copy()
        cost = 0.0
        for k in range(self.horizon):
            u = -self.K @ x
            cost += x @ self.Q @ x + u @ self.R @ u
            x = self.A @ x + self.B @ u
        cost += x @ self.Q @ x  # terminal
        return cost

    def rollout_cost_batch(self, U_batch: torch.Tensor) -> torch.Tensor:
        """Batch rollout on GPU/CPU. U_batch: (N, H, nu) -> costs: (N,).

        Applies process noise and model mismatch if configured.
        """
        device = U_batch.device
        # If model_mismatch > 0, the "true" dynamics differ from nominal
        # Optimizer plans with nominal A/B, but evaluation uses perturbed A'/B'
        mm = self.model_mismatch
        A_true = self.A * (1.0 + mm)  # perturbed
        B_true = self.B * (1.0 - mm)  # perturbed (opposite direction)

        A = torch.tensor(A_true, dtype=torch.float32, device=device)
        B = torch.tensor(B_true, dtype=torch.float32, device=device)
        Q = torch.tensor(self.Q, dtype=torch.float32, device=device)
        R = torch.tensor(self.R, dtype=torch.float32, device=device)
        x0 = torch.tensor(self.x0, dtype=torch.float32, device=device)

        N = U_batch.shape[0]
        x = x0.unsqueeze(0).expand(N, -1).clone()  # (N, nx)
        costs = torch.zeros(N, device=device)

        for k in range(self.horizon):
            u = U_batch[:, k, :]  # (N, nu)
            state_cost = (x @ Q * x).sum(dim=1)
            ctrl_cost = (u @ R * u).sum(dim=1)

            # Non-quadratic: add sinusoidal bumps
            if self.nonquadratic:
                # Creates local minima in action space
                bump = 2.0 * torch.sin(3.0 * u).sum(dim=1)
                ctrl_cost = ctrl_cost + bump

            costs += state_cost + ctrl_cost
            x = x @ A.T + u @ B.T

            # Process noise
            if self.process_noise_std > 0:
                x = x + self.process_noise_std * torch.randn_like(x)

        costs += (x @ Q * x).sum(dim=1)  # terminal
        return costs


# ============================================================
# Optimizers (standalone, no SPIDER dependency)
# ============================================================

@dataclass
class OptimizerConfig:
    num_samples: int = 512
    max_iters: int = 64
    sigma0: float = 0.5
    temperature: float = 1.0
    device: str = "cpu"


def run_pure_mppi(problem: LQRProblem, cfg: OptimizerConfig, seed: int = 0) -> list[float]:
    """Pure MPPI with fixed isotropic covariance."""
    torch.manual_seed(seed)
    device = cfg.device
    H, nu = problem.horizon, problem.nu
    d = H * nu

    mu = torch.zeros(d, device=device)
    sigma = cfg.sigma0
    costs_history = []

    for it in range(cfg.max_iters):
        # Sample
        eps = torch.randn(cfg.num_samples, d, device=device) * sigma
        U = mu.unsqueeze(0) + eps  # (N, d)
        U[0] = mu  # exploit

        # Evaluate
        U_shaped = U.reshape(cfg.num_samples, H, nu)
        costs = problem.rollout_cost_batch(U_shaped)  # (N,)

        # Record exploit cost
        costs_history.append(float(costs[0]))

        # MPPI weights
        J_min = costs.min()
        w = torch.exp(-1.0 / cfg.temperature * (costs - J_min))
        w = w / w.sum()

        # Update mean (no covariance update)
        mu = (w[:, None] * U).sum(dim=0)

    return costs_history


def run_dial_mpc(problem: LQRProblem, cfg: OptimizerConfig, seed: int = 0,
                 beta: float = 0.9) -> list[float]:
    """DIAL-MPC: MPPI with exponential noise annealing (σ decays each iter)."""
    torch.manual_seed(seed)
    device = cfg.device
    H, nu = problem.horizon, problem.nu
    d = H * nu

    mu = torch.zeros(d, device=device)
    sigma = cfg.sigma0
    costs_history = []

    for it in range(cfg.max_iters):
        # Annealed noise scale
        noise_scale = sigma * (beta ** it)

        # Sample
        eps = torch.randn(cfg.num_samples, d, device=device) * noise_scale
        U = mu.unsqueeze(0) + eps
        U[0] = mu

        # Evaluate
        U_shaped = U.reshape(cfg.num_samples, H, nu)
        costs = problem.rollout_cost_batch(U_shaped)

        costs_history.append(float(costs[0]))

        # MPPI weights
        J_min = costs.min()
        w = torch.exp(-1.0 / cfg.temperature * (costs - J_min))
        w = w / w.sum()

        # Update mean
        mu = (w[:, None] * U).sum(dim=0)

    return costs_history


def run_cma_rank(problem: LQRProblem, cfg: OptimizerConfig, seed: int = 0,
                 elite_ratio: float = 0.5) -> list[float]:
    """Hansen's CMA-ES (full covariance, rank-based). Mirrors cma_full.py logic.

    Args:
        elite_ratio: fraction of population to use as parents (default 0.5 = Hansen's default)
    """
    torch.manual_seed(seed)
    device = cfg.device
    H, nu = problem.horizon, problem.nu
    d = H * nu
    lam = cfg.num_samples

    # --- Init (Hansen 2016 defaults, but with configurable mu) ---
    mu_sel = max(1, int(lam * elite_ratio))
    raw_w = torch.log(torch.tensor(mu_sel + 0.5, device=device)) - \
            torch.log(torch.arange(1, mu_sel + 1, device=device, dtype=torch.float32))
    weights = raw_w / raw_w.sum()
    mu_eff = float(1.0 / (weights ** 2).sum())

    c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1) / (d + 1)) - 1.0) + c_sigma
    cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
    c1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2.0 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2) ** 2 + mu_eff))
    chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

    mean = torch.zeros(d, device=device)
    sigma = cfg.sigma0
    C = torch.eye(d, device=device)
    B = torch.eye(d, device=device)
    D = torch.ones(d, device=device)
    invsqrtC = torch.eye(d, device=device)
    p_sigma = torch.zeros(d, device=device)
    p_c = torch.zeros(d, device=device)

    costs_history = []
    eigen_eval = 0

    for gen in range(cfg.max_iters):
        # Sample: x_k = mean + sigma * B * (D .* z_k)
        z = torch.randn(lam, d, device=device)
        y = (z * D.unsqueeze(0)) @ B.T  # (lam, d)
        x = mean.unsqueeze(0) + sigma * y

        # Exploit = mean
        x[0] = mean
        y[0] = 0.0

        # Evaluate
        U_shaped = x.reshape(lam, H, nu)
        costs = problem.rollout_cost_batch(U_shaped)

        costs_history.append(float(costs[0]))

        # Sort (ascending cost = best first for minimization)
        sorted_idx = torch.argsort(costs)  # ascending
        elite_idx = sorted_idx[:mu_sel]

        # Weighted recombination
        y_sel = y[elite_idx]
        y_w = (weights[:, None] * y_sel).sum(dim=0)

        # Update mean
        mean_old = mean.clone()
        mean = mean + sigma * y_w

        # Update p_sigma
        p_sigma = (1 - c_sigma) * p_sigma + \
                  math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrtC @ y_w)

        # Update sigma
        ps_norm = float(p_sigma.norm())
        sigma = sigma * math.exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0))
        sigma = max(1e-20, min(sigma, 1e6))

        # h_sigma
        threshold = (1.4 + 2.0 / (d + 1)) * chi_n * math.sqrt(
            1 - (1 - c_sigma) ** (2 * (gen + 1))
        )
        h_sigma = 1.0 if ps_norm < threshold else 0.0

        # Update p_c
        p_c = (1 - cc) * p_c + h_sigma * math.sqrt(cc * (2 - cc) * mu_eff) * y_w

        # Update C
        delta_h = (1 - h_sigma) * cc * (2 - cc)
        rank_one = p_c.unsqueeze(1) @ p_c.unsqueeze(0)
        sqrt_w = weights.sqrt()
        wy = sqrt_w[:, None] * y_sel
        rank_mu_mat = wy.T @ wy
        C = (1 - c1 - c_mu) * C + c1 * (rank_one + delta_h * C) + c_mu * rank_mu_mat

        # Eigendecomposition (amortized)
        eigen_interval = max(1, int(lam / (c1 + c_mu) / d / 10))
        if (gen + 1) - eigen_eval >= eigen_interval:
            C = 0.5 * (C + C.T)
            try:
                eigenvalues, B = torch.linalg.eigh(C)
            except torch.linalg.LinAlgError:
                C = C + 1e-6 * torch.eye(d, device=device)
                eigenvalues, B = torch.linalg.eigh(C)
            eigenvalues = eigenvalues.clamp(min=1e-20)
            D = eigenvalues.sqrt()
            invsqrtC = B @ torch.diag(1.0 / D) @ B.T
            eigen_eval = gen + 1

    return costs_history


def run_mppi_cma(problem: LQRProblem, cfg: OptimizerConfig, seed: int = 0,
                 eta_sigma: float = 0.3, eta_mu: float = 0.5,
                 mean_update: str = "mppi") -> list[float]:
    """MPPI-CMA (full covariance). Mirrors mppi_cma_full.py logic."""
    torch.manual_seed(seed)
    device = cfg.device
    H, nu = problem.horizon, problem.nu
    d = H * nu
    N = cfg.num_samples
    jitter = 1e-4

    mu = torch.zeros(d, device=device)
    Sigma = (cfg.sigma0 ** 2) * torch.eye(d, device=device)
    costs_history = []

    for it in range(cfg.max_iters):
        # Cholesky
        Sigma_sym = 0.5 * (Sigma + Sigma.T)
        try:
            L = torch.linalg.cholesky(Sigma_sym + jitter * torch.eye(d, device=device))
        except torch.linalg.LinAlgError:
            L = torch.diag(Sigma_sym.diag().clamp(min=jitter).sqrt())

        # Sample eps ~ N(0, Sigma)
        z = torch.randn(N, d, device=device)
        eps = z @ L.T  # (N, d)

        # Candidates
        U = mu.unsqueeze(0) + eps
        U[0] = mu
        eps[0] = 0.0

        # Evaluate
        U_shaped = U.reshape(N, H, nu)
        costs = problem.rollout_cost_batch(U_shaped)

        costs_history.append(float(costs[0]))

        if mean_update == "rank":
            mu_sel = N // 2
            sorted_idx = torch.argsort(costs)  # ascending cost
            selected_idx = sorted_idx[:mu_sel]
            raw_w = torch.log(torch.tensor(mu_sel + 0.5, device=device)) - \
                    torch.log(torch.arange(1, mu_sel + 1, device=device, dtype=torch.float32))
            w_sel = raw_w / raw_w.sum()

            eps_selected = eps[selected_idx]
            weighted_eps_mean = (w_sel[:, None] * eps_selected).sum(dim=0)
            mu_new = mu + weighted_eps_mean

            sqrt_w_sel = w_sel.sqrt()
            weighted_eps = sqrt_w_sel[:, None] * eps_selected
            Sigma_sample = weighted_eps.T @ weighted_eps
        else:
            # MPPI softmax weights
            J_min = costs.min()
            w = torch.exp(-1.0 / cfg.temperature * (costs - J_min))
            w = w / w.sum()

            weighted_eps_mean = (w[:, None] * eps).sum(dim=0)
            mu_new = mu + weighted_eps_mean

            sqrt_w = w.sqrt()
            weighted_eps = sqrt_w[:, None] * eps
            Sigma_sample = weighted_eps.T @ weighted_eps

        # Update
        mu = (1 - eta_mu) * mu + eta_mu * mu_new
        Sigma = (1 - eta_sigma) * Sigma + eta_sigma * Sigma_sample + jitter * torch.eye(d, device=device)
        Sigma = 0.5 * (Sigma + Sigma.T)

    return costs_history


# ============================================================
# Main
# ============================================================

def main():
    scenarios = {
        "Nominal": LQRProblem(nx=4, nu=2, horizon=20),
        "Process Noise (σ=0.05)": LQRProblem(nx=4, nu=2, horizon=20, process_noise_std=0.05),
        "Model Mismatch (20%)": LQRProblem(nx=4, nu=2, horizon=20, model_mismatch=0.2),
        "Non-Quadratic Cost": LQRProblem(nx=4, nu=2, horizon=20, nonquadratic=True),
    }

    cfg = OptimizerConfig(
        num_samples=512,
        max_iters=64,
        sigma0=0.5,
        temperature=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Device: {cfg.device}, Samples: {cfg.num_samples}, Iters: {cfg.max_iters}")

    num_seeds = 5

    optimizers = {
        "Pure MPPI": lambda s, p: run_pure_mppi(p, cfg, seed=s),
        "DIAL-MPC (β=0.9)": lambda s, p: run_dial_mpc(p, cfg, seed=s, beta=0.9),
        "CMA-ES (rank, 50%)": lambda s, p: run_cma_rank(p, cfg, seed=s, elite_ratio=0.5),
        "CMA-ES (rank, 10%)": lambda s, p: run_cma_rank(p, cfg, seed=s, elite_ratio=0.1),
        "MPPI-CMA (rank)": lambda s, p: run_mppi_cma(p, cfg, seed=s, mean_update="rank"),
    }

    colors = {
        "Pure MPPI": "tab:blue",
        "DIAL-MPC (β=0.9)": "tab:orange",
        "CMA-ES (rank, 50%)": "tab:green",
        "CMA-ES (rank, 10%)": "darkgreen",
        "MPPI-CMA (rank)": "tab:red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for sc_idx, (sc_name, problem) in enumerate(scenarios.items()):
        ax = axes_flat[sc_idx]
        optimal_cost = problem.optimal_cost()
        print(f"\n{'='*60}")
        print(f"Scenario: {sc_name}")
        print(f"  Optimal LQR cost (nominal DARE): {optimal_cost:.4f}")

        for opt_name, fn in optimizers.items():
            all_runs = []
            for s in range(num_seeds):
                costs = fn(s, problem)
                all_runs.append(costs)
            data = np.array(all_runs)  # (num_seeds, max_iters)
            final_mean = data[:, -1].mean()
            final_std = data[:, -1].std()
            gap = (final_mean - optimal_cost) / abs(optimal_cost) * 100
            print(f"  {opt_name:25s}: final={final_mean:.4f} ± {final_std:.4f}  (gap={gap:+.2f}%)")

            mean_curve = data.mean(axis=0)
            std_curve = data.std(axis=0)
            iters = np.arange(1, len(mean_curve) + 1)
            ax.plot(iters, mean_curve, label=opt_name, color=colors[opt_name], linewidth=2)
            ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.15, color=colors[opt_name])

        ax.axhline(optimal_cost, color="black", linestyle="--", linewidth=1.5,
                   label=f"Optimal ({optimal_cost:.1f})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost (exploit)")
        ax.set_title(sc_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("LQR Robustness Benchmark — 512 samples, 64 iters, 5 seeds", fontsize=13)
    plt.tight_layout()
    out_path = "/home/roy/.openclaw/workspace/spider/experiments/lqr_benchmark_robust.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
