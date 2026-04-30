"""Unified MPPI / MPPI-CMA optimizer in knot-point space.

Both MPPI and MPPI-CMA share the same initial mean μ₀ and sampling space.
The ONLY difference is whether the covariance adapts:
  - eta_sigma = 0  →  pure MPPI (fixed covariance, equivalent to original SPIDER)
  - eta_sigma > 0  →  MPPI-CMA (Algorithm 2 from the paper)

This ensures a fair comparison: identical starting point, identical noise
structure, identical rollout — only the update rule differs.
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


def make_optimize_once_fn_unified(rollout):
    """Return a single-iteration unified optimizer step.

    Works for both MPPI and MPPI-CMA depending on eta_sigma in cma_state.
    """

    def optimize_once_unified(
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
        lam = config.num_samples  # N
        d = num_knots * config.nu  # total dimension (flattened knot space)
        global_noise_scale = sample_params.get("global_noise_scale", 1.0)

        mu = cma_state["mean"]          # (num_knots, nu)
        sigma = cma_state["sigma"]      # (d,) diagonal std-dev vector
        eta_mu = cma_state["eta_mu"]
        eta_sigma = cma_state["eta_sigma"]  # 0 = pure MPPI, >0 = CMA
        jitter = cma_state.get("jitter", 1e-6)
        temperature = config.temperature

        mu_flat = mu.reshape(-1)  # (d,)

        # ---- 1. Sample perturbations eps ~ N(0, diag(sigma^2)) ----
        eps = torch.randn(lam, d, device=device) * sigma.unsqueeze(0) * global_noise_scale

        # Form candidates u = mu + eps
        u = mu_flat.unsqueeze(0) + eps  # (N, d)

        # Keep first sample as the mean (exploit)
        eps[0] = 0.0
        u[0] = mu_flat

        # Reshape to knot space and interpolate to horizon
        knot_samples = u.reshape(lam, num_knots, config.nu)
        ctrls_samples = interp(knot_samples, config.knot_steps)  # (N, H, nu)

        # ---- 2. Rollout ----
        ctrls_samples, rews, terminate, rollout_info = rollout(
            config, env, ctrls_samples, ref_slice,
            env_params[0] if env_params else {}
        )

        # ---- 3. Compute MPPI weights (Algorithm 2, lines 5-7) ----
        costs = -rews  # (N,)
        J_min = costs.min()
        w_unnorm = torch.exp(-1.0 / temperature * (costs - J_min))
        w = w_unnorm / w_unnorm.sum()  # (N,)

        # ---- 4. Covariance update (line 9): uses old mean ----
        if eta_sigma > 0:
            # MPPI-CMA: adapt covariance
            weighted_eps_sq = (w[:, None] * eps.pow(2)).sum(dim=0)  # (d,)
            sigma_sq_new = (1 - eta_sigma) * sigma.pow(2) + eta_sigma * weighted_eps_sq + jitter
            sigma_new = sigma_sq_new.sqrt()
        else:
            # Pure MPPI: covariance stays fixed
            sigma_new = sigma

        # ---- 5. Mean update (line 12): MPPI weighted mean ----
        mu_new_flat = (1 - eta_mu) * mu_flat + eta_mu * (w[:, None] * u).sum(dim=0)

        # ---- Persist state ----
        cma_state["mean"] = mu_new_flat.reshape(num_knots, config.nu)
        cma_state["sigma"] = sigma_new
        cma_state["generation"] = cma_state.get("generation", 0) + 1

        # ---- Reconstruct best ctrls from updated mean ----
        ctrls_mean = interp(
            cma_state["mean"].unsqueeze(0), config.knot_steps
        ).squeeze(0)

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
        info["rew_max"] = float(rews_np.max())
        info["rew_min"] = float(rews_np.min())
        info["rew_median"] = float(np.median(rews_np))
        info["rew_mean"] = float(rews_np.mean())

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

    return optimize_once_unified


def make_optimize_fn_unified(optimize_once_unified):
    """Return the full optimization loop for the unified optimizer."""

    def optimize_unified(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
        eta_sigma: float = 0.0,
        eta_mu: float = 0.5,
    ):
        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu

        # Initialise state — both MPPI and CMA start from the SAME μ₀
        mean_init = _knots_from_ctrls(ctrls, config)
        sigma_init = torch.full(
            (d,),
            getattr(config, "cma_sigma0", 0.3),
            device=device,
            dtype=torch.float32,
        )
        cma_state = {
            "mean": mean_init,
            "sigma": sigma_init,
            "generation": 0,
            "eta_mu": eta_mu,
            "eta_sigma": eta_sigma,
            "jitter": 1e-6,
        }

        # Noise-annealing schedule
        sample_params_list = [
            {"global_noise_scale": config.beta_traj ** i}
            for i in range(config.max_num_iterations)
        ]

        infos = []
        improvement_history: list[float] = []

        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once_unified(
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

    return optimize_unified
