"""CMA-ES (Hansen's (μ/μ_w, λ)-CMA-ES) for SPIDER trajectory optimization.

Full implementation following Hansen (2016) "The CMA Evolution Strategy:
A Tutorial". Operates in knot-point space, interpolates to full horizon
for rollout.

Key features vs the previous simplified version:
  - Evolution paths (p_σ, p_c) for cumulative step-size adaptation
  - Step-size σ adaptation via CSA (path length control)
  - Rank-one update (via p_c) + rank-μ update of covariance matrix C
  - Eigendecomposition of C for sampling (B D B^T factorization)
  - All standard CMA-ES hyperparameters derived from dimension d

Reference: https://arxiv.org/abs/1604.00772 (Hansen, 2016)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from spider.config import Config
from spider.interp import interp


def _knots_from_ctrls(ctrls: torch.Tensor, config: Config) -> torch.Tensor:
    num_knots = int(round(config.horizon / config.knot_dt))
    indices = torch.arange(num_knots, device=ctrls.device) * config.knot_steps
    indices = indices.clamp(max=ctrls.shape[0] - 1).long()
    return ctrls[indices]


def _init_cma_state(d: int, mean: torch.Tensor, sigma0: float,
                    lam: int, device: torch.device, mu_ratio: float = 0.5) -> dict:
    """Initialize all CMA-ES state variables following Hansen's defaults."""

    # --- Selection and recombination ---
    mu = max(1, int(lam * mu_ratio))
    raw_w = torch.log(torch.tensor(mu + 0.5, device=device)) - torch.log(
        torch.arange(1, mu + 1, dtype=torch.float32, device=device)
    )
    weights = raw_w / raw_w.sum()  # (mu,) positive, sum to 1
    mu_eff = float(1.0 / (weights ** 2).sum())  # variance-effective selection mass

    # --- Adaptation parameters (Table 1, Hansen 2016) ---
    # Step-size control
    c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1) / (d + 1)) - 1.0) + c_sigma

    # Covariance matrix adaptation
    cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
    c1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2.0 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2) ** 2 + mu_eff))

    # Expected length of N(0,I) vector
    chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

    # --- State variables ---
    mean_flat = mean.reshape(-1)  # (d,)

    return {
        "mean": mean_flat,
        "sigma": sigma0,
        "C": torch.eye(d, device=device, dtype=torch.float32),
        "p_sigma": torch.zeros(d, device=device, dtype=torch.float32),
        "p_c": torch.zeros(d, device=device, dtype=torch.float32),
        # Eigendecomposition cache: C = B diag(D^2) B^T
        "B": torch.eye(d, device=device, dtype=torch.float32),
        "D": torch.ones(d, device=device, dtype=torch.float32),
        "invsqrtC": torch.eye(d, device=device, dtype=torch.float32),
        # Hyperparameters (frozen after init)
        "mu": mu,
        "weights": weights,
        "mu_eff": mu_eff,
        "c_sigma": c_sigma,
        "d_sigma": d_sigma,
        "cc": cc,
        "c1": c1,
        "c_mu": c_mu,
        "chi_n": chi_n,
        "generation": 0,
        "eigen_eval": 0,  # track when we last did eigendecomposition
    }


def _update_eigen(state: dict, d: int, lam: int):
    """Eigendecompose C and cache B, D, invsqrtC.

    Only called every ~lambda/(c1+c_mu)/d/10 evaluations for O(d^2) amortization.
    """
    C = state["C"]
    # Enforce symmetry
    C = 0.5 * (C + C.T)
    # Clamp tiny negative eigenvalues from numerical drift
    try:
        eigenvalues, B = torch.linalg.eigh(C)
    except torch.linalg.LinAlgError:
        # Fallback: add jitter and retry
        C = C + 1e-6 * torch.eye(C.shape[0], device=C.device)
        eigenvalues, B = torch.linalg.eigh(C)

    eigenvalues = eigenvalues.clamp(min=1e-20)
    D = eigenvalues.sqrt()
    invsqrtC = B @ torch.diag(1.0 / D) @ B.T

    state["C"] = C
    state["B"] = B
    state["D"] = D
    state["invsqrtC"] = invsqrtC


def make_optimize_once_fn_cma_full(rollout):
    """Return a single-iteration CMA-ES step (Hansen's full algorithm)."""

    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
        env_params=None,
        sample_params=None,
        cma_state=None,
    ):
        if env_params is None:
            env_params = [{}]
        if sample_params is None:
            sample_params = {}
        if cma_state is None:
            raise ValueError("cma_state must be provided")

        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        lam = config.num_samples
        d = num_knots * config.nu

        # Unpack state
        mean = cma_state["mean"]           # (d,)
        sigma = cma_state["sigma"]         # scalar
        B = cma_state["B"]                 # (d, d)
        D = cma_state["D"]                 # (d,)
        invsqrtC = cma_state["invsqrtC"]   # (d, d)
        p_sigma = cma_state["p_sigma"]     # (d,)
        p_c = cma_state["p_c"]             # (d,)
        mu = cma_state["mu"]
        weights = cma_state["weights"]     # (mu,)
        mu_eff = cma_state["mu_eff"]
        c_sigma = cma_state["c_sigma"]
        d_sigma = cma_state["d_sigma"]
        cc = cma_state["cc"]
        c1 = cma_state["c1"]
        c_mu = cma_state["c_mu"]
        chi_n = cma_state["chi_n"]
        gen = cma_state["generation"]

        # ---- 1. Sample λ candidates in DELTA space (additive on reference) ----
        # mean represents a delta (offset) in knot space; starts at zero.
        # Actual knot values = ref_knots + mean + sigma * y
        z = torch.randn(lam, d, device=device)  # z_k ~ N(0, I)
        # y_k = B * (D .* z_k), so delta ~ N(mean, σ^2 C)
        y = (z * D.unsqueeze(0)) @ B.T  # (lam, d)
        delta = mean.unsqueeze(0) + sigma * y  # (lam, d) — delta from reference

        # Keep first sample as exploit (unperturbed mean delta)
        delta[0] = mean
        y[0] = 0.0
        z[0] = 0.0

        # Reconstruct: ref_ctrls + interp(delta_knots)
        H = ctrls.shape[0]
        delta_knots = delta.reshape(lam, num_knots, config.nu)
        delta_horizon = interp(delta_knots, config.knot_steps)  # (lam, H, nu)
        ctrls_samples = ctrls.unsqueeze(0).expand(lam, -1, -1).clone() + delta_horizon[:, :H]

        # Exploit (sample 0) with zero delta = original ctrls (no reconstruction loss)
        # (when mean ≈ 0 at start, sample 0 ≈ ctrls)

        # ---- 2. Rollout and evaluate ----
        ctrls_samples, rews, terminate, rollout_info = rollout(
            config, env, ctrls_samples, ref_slice,
            env_params[0] if env_params else {},
        )

        # ---- 3. Sort by fitness (descending reward = ascending cost) ----
        sorted_idx = torch.argsort(rews, descending=True)  # best first
        elite_idx = sorted_idx[:mu]

        # Weighted recombination of y (search steps in C-scaled space)
        y_sel = y[elite_idx]  # (mu, d)
        y_w = (weights[:, None] * y_sel).sum(dim=0)  # (d,) weighted mean step

        # ---- 4. Update mean ----
        mean_old = mean.clone()
        mean_new = mean + sigma * y_w

        # ---- 5. Update evolution path for step-size (p_σ) ----
        # p_σ ← (1 - c_σ) p_σ + √(c_σ(2-c_σ)μ_eff) · C^{-1/2} · y_w
        p_sigma_new = (
            (1 - c_sigma) * p_sigma
            + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrtC @ y_w)
        )

        # ---- 6. Update step-size σ ----
        # σ ← σ · exp((c_σ/d_σ) · (‖p_σ‖/E[‖N(0,I)‖] - 1))
        ps_norm = float(p_sigma_new.norm())
        sigma_new = sigma * math.exp(
            (c_sigma / d_sigma) * (ps_norm / chi_n - 1.0)
        )
        # Clamp sigma to prevent explosion or collapse
        sigma_new = max(1e-20, min(sigma_new, 1e6))

        # ---- 7. Update evolution path for covariance (p_c) ----
        # h_σ indicator: 1 if ‖p_σ‖ < threshold (prevents stalling)
        threshold = (1.4 + 2.0 / (d + 1)) * chi_n * math.sqrt(
            1 - (1 - c_sigma) ** (2 * (gen + 1))
        )
        h_sigma = 1.0 if ps_norm < threshold else 0.0

        p_c_new = (
            (1 - cc) * p_c
            + h_sigma * math.sqrt(cc * (2 - cc) * mu_eff) * y_w
        )

        # ---- 8. Update covariance matrix C ----
        # C ← (1 - c1 - c_μ) C + c1 (p_c p_c^T + δ(h_σ) cc(2-cc) C) + c_μ Σ w_i y_i y_i^T
        delta_h = (1 - h_sigma) * cc * (2 - cc)  # correction when h_sigma=0

        # Rank-one update
        rank_one = p_c_new.unsqueeze(1) @ p_c_new.unsqueeze(0)  # (d, d)

        # Rank-μ update: Σ w_i y_i:λ y_i:λ^T
        # y_sel is (mu, d), weights is (mu,)
        sqrt_w = weights.sqrt()
        wy = sqrt_w[:, None] * y_sel  # (mu, d)
        rank_mu = wy.T @ wy  # (d, d)

        C_new = (
            (1 - c1 - c_mu) * cma_state["C"]
            + c1 * (rank_one + delta_h * cma_state["C"])
            + c_mu * rank_mu
        )

        # ---- 9. Eigendecomposition (amortized) ----
        cma_state["mean"] = mean_new
        cma_state["sigma"] = sigma_new
        cma_state["p_sigma"] = p_sigma_new
        cma_state["p_c"] = p_c_new
        cma_state["C"] = C_new
        cma_state["generation"] = gen + 1

        # Update eigen every ~lam/(c1+c_mu)/d/10 generations
        eigen_interval = max(1, int(lam / (c1 + c_mu) / d / 10))
        if (gen + 1) - cma_state["eigen_eval"] >= eigen_interval:
            _update_eigen(cma_state, d, lam)
            cma_state["eigen_eval"] = gen + 1

        # ---- Reconstruct best ctrls from updated mean (additive) ----
        H = ctrls.shape[0]
        delta_mean_horizon = interp(
            mean_new.reshape(1, num_knots, config.nu), config.knot_steps
        ).squeeze(0)
        ctrls_mean = ctrls[:H] + delta_mean_horizon[:H]

        # ---- Build info dict ----
        rews_np = rews.cpu().numpy()
        info: dict = {}
        for k, v in rollout_info.items():
            if k in ("trace", "trace_sample"):
                continue
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if v.ndim == 1:
                info[k + "_max"] = v.max()
                info[k + "_min"] = v.min()
                info[k + "_median"] = np.median(v)
                info[k + "_mean"] = v.mean()

        info["improvement"] = float(rews_np.max() - rews_np[0])
        info["rew_u0"] = float(rews_np[0])  # exploit (sample 0 = mean delta)
        info["rew_max"] = float(rews_np.max())
        info["rew_min"] = float(rews_np.min())
        info["rew_median"] = float(np.median(rews_np))
        info["rew_mean"] = float(rews_np.mean())
        info["sigma"] = float(sigma_new)

        if "trace" in rollout_info:
            n_uni = max(0, min(config.num_trace_uniform_samples, lam))
            n_topk = max(0, min(config.num_trace_topk_samples, lam))
            idx_uni = (
                torch.linspace(0, lam - 1, steps=n_uni, dtype=torch.long, device=device)
                if n_uni > 0
                else torch.tensor([], dtype=torch.long, device=device)
            )
            idx_top = (
                torch.topk(rews, k=n_topk, largest=True).indices
                if n_topk > 0
                else torch.tensor([], dtype=torch.long, device=device)
            )
            sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()
            info["trace_sample"] = rollout_info["trace"][sel_idx].cpu().numpy()
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        return ctrls_mean, terminate, info

    return optimize_once


def make_optimize_fn_cma_full(optimize_once):
    """Return the full CMA-ES optimisation loop."""

    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
    ):
        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu

        # Initialize CMA-ES state — mean starts at ZERO (delta space)
        # CMA optimizes offsets from reference ctrls, not absolute knot values
        mean_init = torch.zeros(d, device=device)
        sigma0 = getattr(config, "cma_sigma0", 0.3)
        lam = config.num_samples

        cma_state = _init_cma_state(d, mean_init, sigma0, lam, device,
                                     mu_ratio=getattr(config, "cma_mu_ratio", 0.5))

        # NOTE: CMA-ES manages its own step-size σ via CSA.
        # We do NOT use SPIDER's beta_traj annealing for CMA-ES.
        # sample_params is passed but global_noise_scale is ignored
        # (σ adaptation replaces manual annealing).
        sample_params_list = [{}] * config.max_num_iterations

        infos = []
        improvement_history: list[float] = []

        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once(
                config,
                env,
                ctrls,
                ref_slice,
                config.env_params_list[i] if hasattr(config, "env_params_list") else [{}],
                sample_params_list[i],
                cma_state,
            )
            infos.append(info)
            improvement_history.append(info["improvement"])

            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if (
                len(improvement_history) >= config.improvement_check_steps
                and not terminate_early_stopping
            ):
                recent = improvement_history[-config.improvement_check_steps:]
                if all(imp < config.improvement_threshold for imp in recent):
                    break

        # Pad infos
        fake_info = {k: np.zeros_like(v) for k, v in infos[0].items()}
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)

        info_aggregated: dict = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])

        return ctrls, info_aggregated

    return optimize
