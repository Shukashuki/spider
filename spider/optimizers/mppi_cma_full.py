"""MPPI with Full Covariance Matrix Adaptation (MPPI-CMA-Full).

Implements Algorithm 2 from the paper with FULL covariance matrix,
not diagonal approximation. Operates in knot-point space.

Key difference from mppi_cma.py:
  - Σ is a full d×d matrix (not a diagonal vector)
  - Sampling uses Cholesky decomposition: ε = L @ z, where Σ = L L^T
  - Covariance update: Σ_{t+1} = (1-η_Σ)Σ_t + η_Σ Σ_i w_i ε_i ε_i^T + ε I
"""

from __future__ import annotations

import numpy as np
import torch

from spider.config import Config
from spider.interp import interp


def _knots_from_ctrls(ctrls: torch.Tensor, config: Config) -> torch.Tensor:
    """Extract knot-point values from a full-horizon ctrl tensor."""
    num_knots = int(round(config.horizon / config.knot_dt))
    indices = torch.arange(num_knots, device=ctrls.device) * config.knot_steps
    indices = indices.clamp(max=ctrls.shape[0] - 1).long()
    return ctrls[indices]


def make_optimize_once_fn_mppi_cma_full(rollout):
    """Return a single-iteration MPPI-CMA step with full covariance."""

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
        N = config.num_samples
        d = num_knots * config.nu
        global_noise_scale = sample_params.get("global_noise_scale", 1.0)

        mu = cma_state["mean"]           # (num_knots, nu)
        Sigma = cma_state["Sigma"]       # (d, d) full covariance
        temperature = config.temperature
        eta_mu = getattr(config, "mppi_cma_eta_mu", 0.5)
        eta_sigma = getattr(config, "mppi_cma_eta_sigma", 0.3)
        jitter = getattr(config, "mppi_cma_jitter", 1e-4)

        mu_flat = mu.reshape(-1)  # (d,)

        # ---- 1. Cholesky decomposition for sampling ----
        # Σ = L L^T, sample ε = L @ z where z ~ N(0, I)
        # Ensure PSD: symmetrise and add jitter before attempting Cholesky
        Sigma = 0.5 * (Sigma + Sigma.T)
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(
                    Sigma + (jitter * (10 ** attempt)) * torch.eye(d, device=device)
                )
                break
            except torch.linalg.LinAlgError:
                if attempt == 4:
                    # Last resort: fall back to diagonal (sqrt of diag entries)
                    diag_std = Sigma.diag().clamp(min=jitter).sqrt()
                    L = torch.diag(diag_std)
                    break

        # z ~ N(0, I), shape (N, d)
        z = torch.randn(N, d, device=device)
        # ε = z @ L^T (so each row ε_i ~ N(0, Σ))
        eps = (z @ L.T) * global_noise_scale  # (N, d)

        # Reshape eps to knot space and interpolate to horizon
        eps_knots = eps.reshape(N, num_knots, config.nu)
        delta_ctrls = interp(eps_knots, config.knot_steps)  # (N, H, nu)

        # Form candidates: ctrls + delta (like MPPI, but with CMA covariance)
        H = ctrls.shape[0]
        ctrls_samples = ctrls.unsqueeze(0).expand(N, -1, -1).clone() + delta_ctrls[:, :H]

        # Keep first sample as the exploit (unperturbed)
        ctrls_samples[0] = ctrls[:H]

        # ---- 2. Rollout ----
        ctrls_samples, rews, terminate, rollout_info = rollout(
            config, env, ctrls_samples, ref_slice,
            env_params[0] if env_params else {},
        )

        # ---- 3. Compute weights ----
        mean_update_mode = getattr(config, "mppi_cma_mean_update", "mppi")

        if mean_update_mode == "rank":
            # CMA-ES style: rank-based selection (top-μ weighted recombination)
            mu_ratio = getattr(config, "cma_mu_ratio", 0.5)
            mu_sel = max(1, int(N * mu_ratio))
            sorted_idx = torch.argsort(rews, descending=True)  # best first
            selected_idx = sorted_idx[:mu_sel]

            # Log-linear weights (CMA-ES style)
            raw_w = torch.log(torch.tensor(mu_sel + 0.5, device=device)) - \
                    torch.log(torch.arange(1, mu_sel + 1, device=device, dtype=torch.float32))
            w_sel = raw_w / raw_w.sum()  # (mu_sel,)

            # Weighted mean in knot space: mu + weighted_eps
            eps_selected = eps[selected_idx]  # (mu_sel, d)
            weighted_eps_mean = (w_sel[:, None] * eps_selected).sum(dim=0)  # (d,)
            weighted_mean_knot = mu_flat + weighted_eps_mean

            # Covariance from selected
            sqrt_w_sel = w_sel.sqrt()
            weighted_eps = sqrt_w_sel[:, None] * eps_selected
            Sigma_sample = weighted_eps.T @ weighted_eps  # (d, d)

            # Full-horizon exploit: weighted sum of ctrls_samples
            ctrls_selected = ctrls_samples[selected_idx]  # (mu_sel, H, nu)
            ctrls_weighted = (w_sel[:, None, None] * ctrls_selected).sum(dim=0)  # (H, nu)
        else:
            # MPPI style: softmax weights over all samples
            costs = -rews  # (N,)
            J_min = costs.min()
            w_unnorm = torch.exp(-1.0 / temperature * (costs - J_min))
            w = w_unnorm / w_unnorm.sum()  # (N,)

            # Weighted mean in knot space
            weighted_eps_mean = (w[:, None] * eps).sum(dim=0)  # (d,)
            weighted_mean_knot = mu_flat + weighted_eps_mean

            # Covariance
            sqrt_w = w.sqrt()  # (N,)
            weighted_eps = sqrt_w[:, None] * eps  # (N, d)
            Sigma_sample = weighted_eps.T @ weighted_eps  # (d, d)

            # Full-horizon exploit: weighted sum of ctrls_samples
            ctrls_weighted = (w[:, None, None] * ctrls_samples).sum(dim=0)  # (H, nu)

        # ---- 4. Full covariance update ----
        Sigma_new = (
            (1 - eta_sigma) * Sigma
            + eta_sigma * Sigma_sample
            + jitter * torch.eye(d, device=device)
        )

        # Enforce symmetry
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)

        # ---- 5. Mean update (knot space, for covariance center) ----
        mu_new_flat = (1 - eta_mu) * mu_flat + eta_mu * weighted_mean_knot

        # ---- 6. Full-horizon exploit update ----
        H = ctrls_samples.shape[1]
        ctrls_out = (1 - eta_mu) * ctrls[:H] + eta_mu * ctrls_weighted

        # ---- Persist state ----
        cma_state["mean"] = mu_new_flat.reshape(num_knots, config.nu)
        cma_state["Sigma"] = Sigma_new
        cma_state["generation"] = cma_state.get("generation", 0) + 1

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
        info["rew_u0"] = float(rews_np[0])
        info["rew_max"] = float(rews_np.max())
        info["rew_min"] = float(rews_np.min())
        info["rew_median"] = float(np.median(rews_np))
        info["rew_mean"] = float(rews_np.mean())

        if "trace" in rollout_info:
            n_uni = max(0, min(config.num_trace_uniform_samples, N))
            n_topk = max(0, min(config.num_trace_topk_samples, N))
            idx_uni = (
                torch.linspace(0, N - 1, steps=n_uni, dtype=torch.long, device=device)
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

        return ctrls_out, terminate, info

    return optimize_once


def make_optimize_fn_mppi_cma_full(optimize_once):
    """Return the full MPPI-CMA (full covariance) optimisation loop."""

    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
        eta_sigma: float = 0.3,
    ):
        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu

        # Initialise CMA state with full covariance
        mean_init = _knots_from_ctrls(ctrls, config)
        sigma0 = getattr(config, "cma_sigma0", 0.3)
        Sigma_init = (sigma0 ** 2) * torch.eye(d, device=device, dtype=torch.float32)

        cma_state = {
            "mean": mean_init,
            "Sigma": Sigma_init,
            "generation": 0,
        }

        # Store eta_sigma in config for optimize_once to use
        if not hasattr(config, "mppi_cma_eta_sigma"):
            config.mppi_cma_eta_sigma = eta_sigma
        else:
            config.mppi_cma_eta_sigma = eta_sigma

        # Noise-annealing schedule
        sample_params_list = [
            {"global_noise_scale": config.beta_traj ** i}
            for i in range(config.max_num_iterations)
        ]

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
