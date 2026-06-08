"""Finite-Horizon LQR Algorithm Testbed.

Clean, minimal benchmark for comparing sampling-based optimizers
on a finite-horizon LQR problem with known optimal solution (DARE).

Design:
  - Problem is simple (2D double integrator, d = H×m = 20×2 = 40)
  - Algorithms are PLUGGABLE: implement `Optimizer` protocol, drop in
  - Each optimizer gets: cost_fn, d, N, device → runs T iterations → returns cost history
  - No SPIDER dependencies. Pure torch + scipy.

Usage:
  python lqr_testbed.py                    # run all registered optimizers
  python lqr_testbed.py --only mppi cma    # run subset
  python lqr_testbed.py --sweep eta_mu     # parameter sweep mode
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are


# ============================================================
# Problem
# ============================================================

@dataclass
class LQRProblem:
    """Finite-horizon discrete-time LQR with known optimal cost."""
    nx: int = 4
    nu: int = 2
    H: int = 20
    dt: float = 0.1

    def __post_init__(self):
        # Double integrator: [x, y, vx, vy]
        self.A = np.eye(self.nx)
        self.A[0, 2] = self.dt
        self.A[1, 3] = self.dt
        self.B = np.zeros((self.nx, self.nu))
        self.B[2, 0] = self.dt
        self.B[3, 1] = self.dt
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])
        self.R = np.eye(self.nu) * 0.1
        self.x0 = np.array([1.0, -0.5, 0.2, -0.1])

        # Optimal via DARE
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ (self.B.T @ self.P @ self.A)

    @property
    def d(self) -> int:
        return self.H * self.nu

    def optimal_cost(self) -> float:
        x = self.x0.copy()
        cost = 0.0
        for _ in range(self.H):
            u = -self.K @ x
            cost += x @ self.Q @ x + u @ self.R @ u
            x = self.A @ x + self.B @ u
        cost += x @ self.Q @ x
        return float(cost)

    def make_cost_fn(self, device: str = "cpu") -> Callable[[torch.Tensor], torch.Tensor]:
        """Return batched cost function: (N, d) -> (N,) costs."""
        A = torch.tensor(self.A, dtype=torch.float32, device=device)
        B = torch.tensor(self.B, dtype=torch.float32, device=device)
        Q = torch.tensor(self.Q, dtype=torch.float32, device=device)
        R = torch.tensor(self.R, dtype=torch.float32, device=device)
        x0 = torch.tensor(self.x0, dtype=torch.float32, device=device)
        H, nu = self.H, self.nu

        def cost_fn(U_flat: torch.Tensor) -> torch.Tensor:
            """U_flat: (N, d) -> costs: (N,)"""
            N = U_flat.shape[0]
            U = U_flat.reshape(N, H, nu)
            x = x0.unsqueeze(0).expand(N, -1).clone()
            costs = torch.zeros(N, device=U_flat.device)
            for k in range(H):
                u = U[:, k, :]
                costs += (x @ Q * x).sum(1) + (u @ R * u).sum(1)
                x = x @ A.T + u @ B.T
            costs += (x @ Q * x).sum(1)
            return costs

        return cost_fn


# ============================================================
# Optimizer Protocol
# ============================================================

class Optimizer(ABC):
    """Base class for pluggable optimizers."""

    name: str = "unnamed"

    @abstractmethod
    def reset(self, d: int, device: str):
        """Reset internal state for a new run."""
        ...

    @abstractmethod
    def step(self, cost_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
        """Run one iteration. Return exploit cost (cost of current best guess)."""
        ...

    def get_solution(self) -> torch.Tensor:
        """Return current best solution (d,)."""
        return self.mu.clone()


# ============================================================
# Optimizers
# ============================================================

class PureMPPI(Optimizer):
    """Standard MPPI: full-N softmax weights, fixed isotropic noise."""

    def __init__(self, N: int = 1024, sigma: float = 0.5, temperature: float = 1.0):
        self.name = f"Pure MPPI (σ={sigma})"
        self.N = N
        self.sigma = sigma
        self.temperature = temperature

    def reset(self, d, device):
        self.d = d
        self.device = device
        self.mu = torch.zeros(d, device=device)

    def step(self, cost_fn):
        eps = torch.randn(self.N, self.d, device=self.device) * self.sigma
        U = self.mu.unsqueeze(0) + eps
        U[0] = self.mu; eps[0] = 0.0
        costs = cost_fn(U)
        exploit_cost = float(costs[0])

        # Softmax weights
        c_min = costs.min()
        w = torch.exp(-(costs - c_min) / self.temperature)
        w = w / w.sum()
        self.mu = (w[:, None] * U).sum(0)
        return exploit_cost


class AnnealingMPPI(Optimizer):
    """DIAL-MPC style: softmax weights + geometric noise annealing."""

    def __init__(self, N: int = 1024, sigma0: float = 0.5, beta: float = 0.9,
                 temperature: float = 1.0):
        self.name = f"Annealing MPPI (β={beta})"
        self.N = N
        self.sigma0 = sigma0
        self.beta = beta
        self.temperature = temperature

    def reset(self, d, device):
        self.d = d
        self.device = device
        self.mu = torch.zeros(d, device=device)
        self.iter = 0

    def step(self, cost_fn):
        sigma = self.sigma0 * (self.beta ** self.iter)
        self.iter += 1

        eps = torch.randn(self.N, self.d, device=self.device) * sigma
        U = self.mu.unsqueeze(0) + eps
        U[0] = self.mu; eps[0] = 0.0
        costs = cost_fn(U)
        exploit_cost = float(costs[0])

        c_min = costs.min()
        w = torch.exp(-(costs - c_min) / self.temperature)
        w = w / w.sum()
        self.mu = (w[:, None] * U).sum(0)
        return exploit_cost


class DIALMPPI(Optimizer):
    """DIAL-MPC: top-k% elite selection + softmax + annealing."""

    def __init__(self, N: int = 1024, sigma0: float = 0.5, beta: float = 0.9,
                 temperature: float = 0.1, elite_ratio: float = 0.1):
        self.name = f"DIAL-MPC (β={beta}, top-{int(elite_ratio*100)}%)"
        self.N = N
        self.sigma0 = sigma0
        self.beta = beta
        self.temperature = temperature
        self.elite_ratio = elite_ratio

    def reset(self, d, device):
        self.d = d
        self.device = device
        self.mu = torch.zeros(d, device=device)
        self.iter = 0

    def step(self, cost_fn):
        sigma = self.sigma0 * (self.beta ** self.iter)
        self.iter += 1

        eps = torch.randn(self.N, self.d, device=self.device) * sigma
        U = self.mu.unsqueeze(0) + eps
        U[0] = self.mu; eps[0] = 0.0
        costs = cost_fn(U)
        exploit_cost = float(costs[0])

        # Top-k elite selection
        rews = -costs
        top_k = max(1, int(self.elite_ratio * self.N))
        top_idx = torch.topk(rews, k=top_k, largest=True).indices
        top_rews = rews[top_idx]
        top_rews_norm = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
        w = torch.zeros_like(costs)
        w[top_idx] = torch.softmax(top_rews_norm / self.temperature, dim=0)
        self.mu = (w[:, None] * U).sum(0)
        return exploit_cost


class MPPICMA(Optimizer):
    """MPPI-CMA: full covariance adaptation with configurable weight scheme."""

    def __init__(self, N: int = 1024, sigma0: float = 0.5,
                 eta_mu: float = 0.5, eta_sigma: float = 0.3,
                 temperature: float = 1.0,
                 weight_mode: str = "mppi",  # "mppi" | "rank" | "dial"
                 elite_ratio: float = 0.5):
        self.name = f"MPPI-CMA (η_μ={eta_mu}, η_Σ={eta_sigma}, w={weight_mode})"
        self.N = N
        self.sigma0 = sigma0
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma
        self.temperature = temperature
        self.weight_mode = weight_mode
        self.elite_ratio = elite_ratio
        self.jitter = 1e-4

    def reset(self, d, device):
        self.d = d
        self.device = device
        self.mu = torch.zeros(d, device=device)
        self.Sigma = (self.sigma0 ** 2) * torch.eye(d, device=device)

    def _cholesky(self):
        S = 0.5 * (self.Sigma + self.Sigma.T)
        for attempt in range(5):
            try:
                return torch.linalg.cholesky(
                    S + (self.jitter * (10 ** attempt)) * torch.eye(self.d, device=self.device))
            except torch.linalg.LinAlgError:
                pass
        return torch.diag(S.diag().clamp(min=self.jitter).sqrt())

    def _weights(self, costs):
        if self.weight_mode == "mppi":
            c_min = costs.min()
            w = torch.exp(-(costs - c_min) / self.temperature)
            return w / w.sum()
        elif self.weight_mode == "rank":
            mu_sel = max(1, int(self.N * self.elite_ratio))
            sorted_idx = torch.argsort(costs)  # ascending
            raw_w = torch.log(torch.tensor(mu_sel + 0.5, device=self.device)) - \
                    torch.log(torch.arange(1, mu_sel + 1, device=self.device, dtype=torch.float32))
            w = torch.zeros_like(costs)
            w[sorted_idx[:mu_sel]] = raw_w / raw_w.sum()
            return w
        elif self.weight_mode == "dial":
            rews = -costs
            top_k = max(1, int(0.1 * self.N))
            top_idx = torch.topk(rews, k=top_k, largest=True).indices
            top_rews = rews[top_idx]
            top_rews_norm = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
            w = torch.zeros_like(costs)
            w[top_idx] = torch.softmax(top_rews_norm / self.temperature, dim=0)
            return w
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

    def step(self, cost_fn):
        L = self._cholesky()
        z = torch.randn(self.N, self.d, device=self.device)
        eps = z @ L.T
        U = self.mu.unsqueeze(0) + eps
        U[0] = self.mu; eps[0] = 0.0
        costs = cost_fn(U)
        exploit_cost = float(costs[0])

        w = self._weights(costs)

        # Covariance update (before mean)
        sqrt_w = w.sqrt()
        we = sqrt_w[:, None] * eps
        Sigma_sample = we.T @ we
        self.Sigma = (1 - self.eta_sigma) * self.Sigma + self.eta_sigma * Sigma_sample \
                     + self.jitter * torch.eye(self.d, device=self.device)
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        # Mean update (EMA)
        weighted_eps = (w[:, None] * eps).sum(0)
        mu_new = self.mu + weighted_eps
        self.mu = (1 - self.eta_mu) * self.mu + self.eta_mu * mu_new

        return exploit_cost


# ============================================================
# Runner
# ============================================================

@dataclass
class RunConfig:
    iters: int = 128
    seeds: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_optimizer(opt: Optimizer, problem: LQRProblem, cfg: RunConfig) -> np.ndarray:
    """Run optimizer across seeds. Returns (seeds, iters) cost array."""
    cost_fn = problem.make_cost_fn(cfg.device)
    all_runs = []
    for s in range(cfg.seeds):
        torch.manual_seed(s)
        opt.reset(problem.d, cfg.device)
        costs = []
        for _ in range(cfg.iters):
            c = opt.step(cost_fn)
            costs.append(c)
        all_runs.append(costs)
    return np.array(all_runs)


def run_all(optimizers: list[Optimizer], problem: LQRProblem, cfg: RunConfig):
    """Run all optimizers and return {name: (seeds, iters) array}."""
    results = {}
    opt_cost = problem.optimal_cost()
    print(f"LQR Optimal: {opt_cost:.4f}  |  d={problem.d}  |  device={cfg.device}")
    print(f"{'Optimizer':40s}  {'Final Cost':>12s}  {'Gap':>10s}")
    print("-" * 70)
    for opt in optimizers:
        data = run_optimizer(opt, problem, cfg)
        results[opt.name] = data
        fm, fs = data[:, -1].mean(), data[:, -1].std()
        gap = (fm - opt_cost) / abs(opt_cost) * 100
        print(f"{opt.name:40s}  {fm:10.4f}±{fs:.4f}  {gap:+8.3f}%")
    return results, opt_cost


# ============================================================
# Plotting
# ============================================================

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_convergence(results: dict, opt_cost: float, cfg: RunConfig,
                     title: str = "", save_path: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iters = np.arange(1, cfg.iters + 1)

    for i, (name, data) in enumerate(results.items()):
        m = data.mean(axis=0)
        s = data.std(axis=0)
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        ax.semilogy(iters, np.maximum(m - opt_cost, 1e-8), label=name, color=color, linewidth=2)
        ax.fill_between(iters,
                        np.maximum(m - s - opt_cost, 1e-8),
                        np.maximum(m + s - opt_cost, 1e-8),
                        alpha=0.15, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost Gap  J(u) - J* (log)")
    ax.set_title(title or f"LQR Testbed: d={cfg.iters}, {cfg.seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_sweep(sweep_results: dict, param_name: str, opt_cost: float,
               cfg: RunConfig, save_path: str | None = None):
    """Plot parameter sweep: one line per param value."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iters = np.arange(1, cfg.iters + 1)

    for i, (val, data) in enumerate(sweep_results.items()):
        m = data.mean(axis=0)
        s = data.std(axis=0)
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        ax.semilogy(iters, np.maximum(m - opt_cost, 1e-8),
                    label=f"{param_name}={val}", color=color, linewidth=2)
        ax.fill_between(iters,
                        np.maximum(m - s - opt_cost, 1e-8),
                        np.maximum(m + s - opt_cost, 1e-8),
                        alpha=0.15, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost Gap  J(u) - J* (log)")
    ax.set_title(f"Parameter Sweep: {param_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Preset Experiments
# ============================================================

def default_optimizers() -> list[Optimizer]:
    """Standard 4-way comparison."""
    return [
        PureMPPI(N=1024, sigma=0.5, temperature=1.0),
        AnnealingMPPI(N=1024, sigma0=0.5, beta=0.9, temperature=1.0),
        DIALMPPI(N=1024, sigma0=0.5, beta=0.9, temperature=0.1, elite_ratio=0.1),
        MPPICMA(N=1024, sigma0=0.5, eta_mu=1.0, eta_sigma=0.3,
                temperature=1.0, weight_mode="mppi"),
    ]


def sweep_eta_mu(problem, cfg) -> dict:
    """Sweep η_μ for MPPI-CMA."""
    values = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}
    for v in values:
        opt = MPPICMA(N=1024, sigma0=0.5, eta_mu=v, eta_sigma=0.3,
                      temperature=1.0, weight_mode="mppi")
        data = run_optimizer(opt, problem, cfg)
        results[v] = data
        fm = data[:, -1].mean()
        gap = (fm - problem.optimal_cost()) / abs(problem.optimal_cost()) * 100
        print(f"  η_μ={v:5.2f}: {fm:.4f} (gap={gap:+.3f}%)")
    return results


def sweep_eta_sigma(problem, cfg) -> dict:
    """Sweep η_Σ for MPPI-CMA."""
    values = [0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}
    for v in values:
        opt = MPPICMA(N=1024, sigma0=0.5, eta_mu=0.5, eta_sigma=v,
                      temperature=1.0, weight_mode="mppi")
        data = run_optimizer(opt, problem, cfg)
        results[v] = data
        fm = data[:, -1].mean()
        gap = (fm - problem.optimal_cost()) / abs(problem.optimal_cost()) * 100
        print(f"  η_Σ={v:5.2f}: {fm:.4f} (gap={gap:+.3f}%)")
    return results


def sweep_weight_mode(problem, cfg) -> dict:
    """Compare weight schemes for MPPI-CMA."""
    modes = {"mppi": "mppi", "rank-50%": "rank", "dial-10%": "dial"}
    results = {}
    for label, mode in modes.items():
        opt = MPPICMA(N=1024, sigma0=0.5, eta_mu=1.0, eta_sigma=0.3,
                      temperature=1.0 if mode != "dial" else 0.1,
                      weight_mode=mode)
        data = run_optimizer(opt, problem, cfg)
        results[label] = data
        fm = data[:, -1].mean()
        gap = (fm - problem.optimal_cost()) / abs(problem.optimal_cost()) * 100
        print(f"  {label:12s}: {fm:.4f} (gap={gap:+.3f}%)")
    return results


def sweep_sigma0(problem, cfg) -> dict:
    """Sweep σ₀ for MPPI-CMA."""
    values = [0.05, 0.1, 0.25, 0.5, 1.0]
    results = {}
    for v in values:
        opt = MPPICMA(N=1024, sigma0=v, eta_mu=1.0, eta_sigma=0.3,
                      temperature=1.0, weight_mode="mppi")
        data = run_optimizer(opt, problem, cfg)
        results[v] = data
        fm = data[:, -1].mean()
        gap = (fm - problem.optimal_cost()) / abs(problem.optimal_cost()) * 100
        print(f"  σ₀={v:5.2f}: {fm:.4f} (gap={gap:+.3f}%)")
    return results


# ============================================================
# Main
# ============================================================

SWEEPS = {
    "eta_mu": sweep_eta_mu,
    "eta_sigma": sweep_eta_sigma,
    "weight_mode": sweep_weight_mode,
    "sigma0": sweep_sigma0,
}


def main():
    parser = argparse.ArgumentParser(description="LQR Algorithm Testbed")
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sweep", type=str, default=None, choices=list(SWEEPS.keys()),
                        help="Run parameter sweep instead of default comparison")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only named optimizers (substring match)")
    parser.add_argument("--out", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = LQRProblem()
    cfg = RunConfig(iters=args.iters, seeds=args.seeds, device=device)
    out_dir = "/home/roy/.openclaw/workspace/spider/outputs"

    if args.sweep:
        print(f"\n{'='*60}")
        print(f"Sweep: {args.sweep}")
        print(f"{'='*60}")
        sweep_fn = SWEEPS[args.sweep]
        sweep_results = sweep_fn(problem, cfg)
        save_path = args.out or f"{out_dir}/lqr_testbed_sweep_{args.sweep}.png"
        plot_sweep(sweep_results, args.sweep, problem.optimal_cost(), cfg, save_path)
    else:
        optimizers = default_optimizers()
        if args.only:
            optimizers = [o for o in optimizers
                          if any(s.lower() in o.name.lower() for s in args.only)]

        print(f"\n{'='*60}")
        print(f"LQR Testbed: {len(optimizers)} optimizers, {cfg.iters} iters, {cfg.seeds} seeds")
        print(f"{'='*60}")
        results, opt_cost = run_all(optimizers, problem, cfg)
        save_path = args.out or f"{out_dir}/lqr_testbed_compare.png"
        plot_convergence(results, opt_cost, cfg,
                         title=f"LQR Testbed: {cfg.iters} iters × {cfg.seeds} seeds, N=1024",
                         save_path=save_path)


if __name__ == "__main__":
    main()
