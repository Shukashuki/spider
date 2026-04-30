"""MPPI-CMA in horizon space — drop-in replacement for SPIDER's original MPPI.

Identical to sampling.py's optimize_once EXCEPT:
  - Maintains a diagonal covariance σ in horizon space (H, nu)
  - After computing MPPI weights, updates σ using weighted second moment of perturbations
  - Uses σ to scale the noise (replaces config.noise_scale with adaptive version)

Everything else — sampling via knot interp, rollout, weight computation,
mean update (horizon-space weighted average) — is EXACTLY the same as original SPIDER.
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import torch.nn.functional as F

from spider.config import Config
from spider.interp import interp


def _compute_weights_impl(
    rews: torch.Tensor, num_samples: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute softmax weights — identical to sampling.py."""
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = (
        rews[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1000.0, device=rews.device)
    )
    rews = torch.where(nan_mask, rews_min, rews)

    top_k = max(1, int(0.1 * num_samples))
    top_indices = torch.topk(rews, k=top_k, largest=True).indices

    weights = torch.zeros_like(rews)
    top_rews = rews[top_indices]
    top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
    weights[top_indices] = top_weights

    return weights, nan_mask


def make_optimize_once_fn_cma_horizon(rollout):
    """Single-iteration MPPI-CMA step in horizon space.

    Sampling and rollout are identical to original SPIDER.
    The only addition: covariance adaptation after weight computation.
    """

    def optimize_once_cma_horizon(
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
        eta_sigma = cma_state["eta_sigma"]
        jitter = cma_state.get("jitter", 1e-6)
        global_noise_scale = sample_params.get("global_noise_scale", 1.0)

        # ---- 1. Sample (identical to original SPIDER) ----
        # noise_scale: (num_samples, num_knots, nu) — from config
        # But we modulate it with our adaptive sigma if available
        knot_noise = (
            torch.randn_like(config.noise_scale, device=device)
            * config.noise_scale
            * global_noise_scale
        )
        # interp to horizon
        delta_ctrl_samples = interp(knot_noise, config.knot_steps)  # (N, H, nu)

        # If we have adaptive sigma, apply it as a per-dimension multiplier
        # sigma shape: (H, nu), delta shape: (N, H, nu)
        if "sigma" in cma_state and cma_state["sigma"] is not None:
            sigma = cma_state["sigma"]  # (H, nu)
            # sigma is relative scaling: 1.0 = no change from original noise
            delta_ctrl_samples = delta_ctrl_samples * sigma.unsqueeze(0)

        ctrls_samples = ctrls.unsqueeze(0) + delta_ctrl_samples  # (N, H, nu)

        # Keep first sample as exploit (no noise) — same as original
        ctrls_samples[0] = ctrls

        # ---- 2. Rollout (identical to original SPIDER) ----
        min_rew = torch.full((config.num_samples,), float("inf"), device=device)
        for env_param in env_params:
            ctrls_samples, rews, terminate, rollout_info = rollout(
                config, env, ctrls_samples, ref_slice, env_param,
            )
            min_rew = torch.minimum(min_rew, rews)
        rews = min_rew

        # ---- 3. Compute weights (identical to original SPIDER) ----
        weights, nan_mask = _compute_weights_impl(
            rews, config.num_samples, config.temperature
        )

        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        # ---- 4. Mean update (identical to original SPIDER) ----
        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)  # (H, nu)

        # ---- 5. Covariance adaptation (THE ONLY ADDITION) ----
        # Perturbations relative to old mean (= ctrls before update)
        eps = ctrls_samples - ctrls.unsqueeze(0)  # (N, H, nu)
        # Weighted second moment per dimension
        weighted_eps_sq = (weights[:, None, None] * eps.pow(2)).sum(dim=0)  # (H, nu)

        if "sigma" not in cma_state or cma_state["sigma"] is None:
            # First iteration: initialize sigma from observed variance
            cma_state["sigma"] = weighted_eps_sq.sqrt().clamp(min=jitter)
        else:
            old_sigma = cma_state["sigma"]
            sigma_sq_new = (
                (1 - eta_sigma) * old_sigma.pow(2)
                + eta_sigma * weighted_eps_sq
                + jitter
            )
            cma_state["sigma"] = sigma_sq_new.sqrt()

        cma_state["generation"] = cma_state.get("generation", 0) + 1

        # ---- 6. Info (identical to original SPIDER) ----
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(0, config.num_samples - 1, steps=n_uni,
                           dtype=torch.long, device=device)
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=device)
        )
        idx_top = (
            torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        info = {}
        for k, v in rollout_info.items():
            if k not in ["trace", "trace_sample"]:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if v.ndim == 1:
                    info[k + "_max"] = v.max()
                    info[k + "_min"] = v.min()
                    info[k + "_median"] = np.median(v)
                    info[k + "_mean"] = v.mean()

        rews_np = rews.cpu().numpy()
        info["improvement"] = rews_np.max() - rews_np[0]
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()

        if "trace" in rollout_info:
            info["trace_sample"] = rollout_info["trace"][sel_idx].cpu().numpy()
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        return ctrls_mean, terminate, info

    return optimize_once_cma_horizon


def make_optimize_fn_cma_horizon(optimize_once_cma_horizon):
    """Full optimization loop — identical structure to original SPIDER's make_optimize_fn."""

    def optimize_cma_horizon(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
        eta_sigma: float = 0.3,
    ):
        # Initialize CMA state — sigma starts at None (will be set from first iteration)
        cma_state = {
            "eta_sigma": eta_sigma,
            "sigma": None,  # will be initialized from first iteration's variance
            "generation": 0,
            "jitter": 1e-6,
        }

        # No annealing — CMA adapts its own covariance
        sample_params_list = [
            {"global_noise_scale": 1.0}
            for _ in range(config.max_num_iterations)
        ]

        infos = []
        improvement_history = []

        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once_cma_horizon(
                config,
                env,
                ctrls,
                ref_slice,
                config.env_params_list[i],
                sample_params_list[i],
                cma_state,
            )
            infos.append(info)
            improvement_history.append(info["improvement"])

            # Early stopping — identical to original SPIDER
            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if (
                len(improvement_history) >= config.improvement_check_steps
                and not terminate_early_stopping
            ):
                recent = improvement_history[-config.improvement_check_steps:]
                if all(imp < config.improvement_threshold for imp in recent):
                    break

        # Pad infos — identical to original SPIDER
        fake_info = {k: np.zeros_like(v) for k, v in infos[0].items()}
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)

        info_aggregated = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])

        return ctrls, info_aggregated

    return optimize_cma_horizon
