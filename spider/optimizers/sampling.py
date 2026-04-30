# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define functions to get noise schedule for the optimizer.

Convention:
- All info should be numpy array.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import torch.nn.functional as F

from spider.config import Config
from spider.interp import interp


def _sample_ctrls_impl(
    config, ctrls: torch.Tensor, sample_params: dict | None = None, cma_state=None
) -> torch.Tensor:
    """Sample control actions from the control signal (implementation).

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)
        cma_state: Optional CMA state dict with "Sigma" key for full-covariance sampling

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    if sample_params is None:
        sample_params = {}
    global_noise_scale = sample_params.get("global_noise_scale", 1.0)

    if cma_state is not None and cma_state.get("Sigma") is not None:
        # CMA mode: sample from full covariance in knot space via Cholesky
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu
        N = config.num_samples
        Sigma = cma_state["Sigma"]
        jitter = getattr(config, "mppi_cma_jitter", 1e-4)

        Sigma_symm = 0.5 * (Sigma + Sigma.T)
        L = None
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(
                    Sigma_symm
                    + (jitter * (10**attempt)) * torch.eye(d, device=config.device)
                )
                break
            except torch.linalg.LinAlgError:
                pass
        if L is None:
            L = torch.diag(Sigma_symm.diag().clamp(min=jitter).sqrt())

        z = torch.randn(N, d, device=config.device)
        eps_flat = (z @ L.T) * global_noise_scale  # (N, d)
        eps_knots = eps_flat.reshape(N, num_knots, config.nu)
        delta_ctrl_samples = interp(eps_knots, config.knot_steps)  # (N, H, nu)
        ctrls_samples = ctrls + delta_ctrl_samples  # broadcast (H,nu) + (N,H,nu)

        # Exploit sample (index 0) = unperturbed; zero its eps for covariance update
        ctrls_samples[0] = ctrls
        eps_flat = eps_flat.clone()
        eps_flat[0] = 0.0
        cma_state["_last_eps"] = eps_flat
    else:
        # Original isotropic sampling (DIAL-MPC / Pure MPPI)
        knot_samples = (
            torch.randn_like(config.noise_scale, device=config.device)
            * config.noise_scale
            * global_noise_scale
        )
        delta_ctrl_samples = interp(knot_samples, config.knot_steps)
        ctrls_samples = ctrls + delta_ctrl_samples

    return ctrls_samples


# Compiled version — only used for non-CMA modes (no mutable side effects)
if hasattr(torch, "compile"):
    _sample_ctrls_compiled = torch.compile(_sample_ctrls_impl)
else:
    _sample_ctrls_compiled = _sample_ctrls_impl


def sample_ctrls(
    config, ctrls: torch.Tensor, sample_params: dict | None = None, cma_state=None
) -> torch.Tensor:
    """Sample control actions from the control signal.

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)
        cma_state: Optional CMA state dict; bypasses torch.compile when set

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    if config.use_torch_compile and cma_state is None:
        return _sample_ctrls_compiled(config, ctrls, sample_params, None)
    else:
        return _sample_ctrls_impl(config, ctrls, sample_params, cma_state)


def make_rollout_fn(
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    save_env_params,
    load_env_params,
    copy_sample_state,
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> torch.Tensor:
        """Rollout the control actions to get reward

        Args:
            config: Config
            env: Environment
            ctrls: Control actions, shape (horizon_steps, nu)
            ref_slice: Reference slice, shape (nq, nv, nu, ncon, ncon_pos)

        Returns:
            Reward, shape (num_samples,)
            Info: dict, including trace (N, H, n_trace, 3)
        """
        # save initial state
        init_state = save_state(env)

        # save initial env params (active group pointer)
        init_env_param = save_env_params(config, env)

        # select rollout graph/data pointers for this rollout
        env = load_env_params(config, env, env_param)

        # rollout to get reward
        N, H = ctrls.shape[:2]
        trace_list = []
        cum_rew = torch.zeros(N, device=config.device)
        info_list = []
        for t in range(H):
            # step the environment
            step_env(config, env, ctrls[:, t])  # (N, nu)
            # get reward
            ref = [r[t] for r in ref_slice]
            rew, info = (
                get_reward(config, env, ref)
                if t < H - 1
                else get_terminal_reward(config, env, ref)
            )
            cum_rew += rew
            # get trace
            trace = get_trace(config, env)
            trace_list.append(trace)
            info_list.append(info)
            # Resampling: replace bad samples with good samples periodically
            terminate = get_terminate(config, env, ref)
            if (
                config.terminate_resample
                and t < H - 1
                and terminate.any()
                and (not terminate.all())
            ):
                bad_indices = torch.nonzero(terminate).squeeze(-1)
                good_indices = torch.nonzero(~terminate).squeeze(-1)
                # make sure good indices shape is the same as bad indices
                if good_indices.shape[0] > bad_indices.shape[0]:
                    good_indices = good_indices[: bad_indices.shape[0]]
                elif good_indices.shape[0] < bad_indices.shape[0]:
                    random_idx = torch.randint(
                        0, good_indices.shape[0], (bad_indices.shape[0],)
                    )
                    good_indices = good_indices[random_idx]

                # Replace bad sample simulation state with good sample simulation state
                copy_sample_state(config, env, good_indices, bad_indices)

                # Replace bad samples control with good samples (for initial control and current timestep only)
                ctrls[bad_indices, :t] = ctrls[good_indices, :t]

                # Replace bad samples cumulative reward with good samples reward
                cum_rew[bad_indices] = cum_rew[good_indices]

        info_combined = {
            k: torch.stack([info[k] for info in info_list], axis=0)
            for k in info_list[0].keys()
        }
        mean_info = {k: v.mean(axis=0) for k, v in info_combined.items()}
        mean_rew = cum_rew / H

        # reset all envs back to initial state
        env = load_state(env, init_state)

        # reset env params
        env = load_env_params(config, env, init_env_param)

        # get info
        trace_list = torch.stack(trace_list, dim=1)
        info = {
            "trace": trace_list,  # (N, H, n_trace, 3)
            **mean_info,
        }
        return ctrls, mean_rew, terminate, info

    return rollout


def _compute_weights_impl(
    rews: torch.Tensor, num_samples: int, temperature: float, mode: str = "dial"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weights from rewards (implementation).

    Args:
        rews: Rewards, shape (num_samples,)
        num_samples: Number of samples
        temperature: Temperature for softmax
        mode: "dial" (top-10% softmax), "mppi" (full-N softmax), "cma_rank" (log-linear rank),
              "cma_dial" (top-10% softmax, used with CMA full-covariance sampling)

    Returns:
        Weights, shape (num_samples,)
        nan_mask, shape (num_samples,)
    """
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = (
        rews[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1000.0, device=rews.device)
    )
    rews = torch.where(nan_mask, rews_min, rews)

    if mode == "dial":
        # Original: top-10% elite selection + softmax
        top_k = max(1, int(0.1 * num_samples))
        top_indices = torch.topk(rews, k=top_k, largest=True).indices
        weights = torch.zeros_like(rews)
        top_rews = rews[top_indices]
        top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
        top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
        weights[top_indices] = top_weights

    elif mode == "mppi":
        # Pure MPPI: softmax over all samples
        costs = -rews
        J_min = costs.min()
        w_unnorm = torch.exp(-1.0 / temperature * (costs - J_min))
        weights = w_unnorm / w_unnorm.sum()

    elif mode == "cma_rank":
        # CMA-ES rank-based: top-μ log-linear weights
        mu_ratio = 0.5
        mu_sel = max(1, int(num_samples * mu_ratio))
        sorted_idx = torch.argsort(rews, descending=True)
        selected_idx = sorted_idx[:mu_sel]
        raw_w = torch.log(
            torch.tensor(mu_sel + 0.5, device=rews.device)
        ) - torch.log(
            torch.arange(1, mu_sel + 1, device=rews.device, dtype=torch.float32)
        )
        weights = torch.zeros_like(rews)
        weights[selected_idx] = raw_w / raw_w.sum()

    elif mode == "cma_dial":
        # CMA sampling + DIAL top-10% elite softmax weighting
        top_k = max(1, int(0.1 * num_samples))
        top_indices = torch.topk(rews, k=top_k, largest=True).indices
        weights = torch.zeros_like(rews)
        top_rews = rews[top_indices]
        top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
        top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
        weights[top_indices] = top_weights

    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    return weights, nan_mask


# Compiled version — dial mode only (no control-flow branching on mode)
if hasattr(torch, "compile"):
    _compute_weights_compiled = torch.compile(_compute_weights_impl)
else:
    _compute_weights_compiled = _compute_weights_impl


def make_optimize_once_fn(
    rollout,
):
    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_params: list[dict] = [{}],
        sample_params: dict | None = None,
        cma_state=None,
    ) -> torch.Tensor:
        """Single step optimization of the policy parameters.

        Supports four modes via config.optimizer_mode:
          "dial"     — DIAL-MPC (top-10% elite softmax, annealing via beta_traj)
          "mppi"     — Pure MPPI (full-N softmax, no annealing)
          "cma_rank" — MPPI-CMA with full covariance + rank-based weights
          "cma_dial" — MPPI-CMA full covariance + top-10% elite softmax weights

        Args:
            config: Config
            env: Environment
            ctrls: Control actions, shape (horizon_steps, num_actions)
            ref_slice: Reference slice
            env_params: List of env parameter dicts for domain randomization
            sample_params: Optional sampling parameters (e.g., global_noise_scale)
            cma_state: CMA state dict (required when optimizer_mode="cma_rank")

        Returns:
            Control actions, terminate flag, info dict
        """
        mode = getattr(config, "optimizer_mode", "dial")

        # Sample controls (CMA mode uses Cholesky sampling from cma_state["Sigma"])
        ctrls_samples = sample_ctrls(config, ctrls, sample_params, cma_state)
        # For non-CMA modes, set exploit sample explicitly (CMA sets it in _sample_ctrls_impl)
        if cma_state is None:
            ctrls_samples[0] = ctrls

        # Rollout with domain randomization — take worst-case reward across DR sets
        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        for env_param in env_params:
            ctrls_samples, rews, terminate, rollout_info = rollout(
                config,
                env,
                ctrls_samples,
                ref_slice,
                env_param,
            )
            min_rew = torch.minimum(min_rew, rews)
        rews = min_rew

        # Compute weights (mode-aware; use compiled version only for dial mode)
        weight_mode = {
            "dial": "dial", "mppi": "mppi",
            "cma_rank": "cma_rank", "cma_dial": "cma_dial",
        }.get(mode, "dial")
        if config.use_torch_compile and mode == "dial":
            weights, nan_mask = _compute_weights_compiled(
                rews, config.num_samples, config.temperature
            )
        else:
            weights, nan_mask = _compute_weights_impl(
                rews, config.num_samples, config.temperature, mode=weight_mode
            )

        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)

        # CMA-specific: full covariance EMA update + mean EMA (both cma_rank and cma_dial)
        if mode in ("cma_rank", "cma_dial") and cma_state is not None:
            eps = cma_state.get("_last_eps")  # (N, d), stored by _sample_ctrls_impl
            if eps is not None:
                eta_mu = getattr(config, "mppi_cma_eta_mu", 0.5)
                eta_sigma = getattr(config, "mppi_cma_eta_sigma", 0.3)
                jitter = getattr(config, "mppi_cma_jitter", 1e-4)
                d = eps.shape[1]
                num_knots = int(round(config.horizon / config.knot_dt))

                # Covariance update always uses rank-based top-50% weights for stability.
                # cma_dial uses DIAL top-10% only for the mean/ctrls update below.
                mu_ratio = 0.5
                mu_sel = max(1, int(config.num_samples * mu_ratio))
                sorted_idx = torch.argsort(rews, descending=True)
                selected_idx = sorted_idx[:mu_sel]
                raw_w_sigma = torch.log(
                    torch.tensor(mu_sel + 0.5, device=rews.device)
                ) - torch.log(
                    torch.arange(1, mu_sel + 1, device=rews.device, dtype=torch.float32)
                )
                w_sigma = torch.zeros_like(rews)
                w_sigma[selected_idx] = raw_w_sigma / raw_w_sigma.sum()

                sqrt_w_sigma = w_sigma.sqrt()
                weighted_eps = sqrt_w_sigma[:, None] * eps  # (N, d)
                Sigma_sample = weighted_eps.T @ weighted_eps  # (d, d)
                Sigma = cma_state["Sigma"]
                # Clip eigenvalues: prevent unbounded growth
                sigma_max = float(config.noise_scale[1].max().item() ** 2) * 100.0
                Sigma_new = (
                    (1 - eta_sigma) * Sigma
                    + eta_sigma * Sigma_sample
                    + jitter * torch.eye(d, device=config.device)
                )
                Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
                diag = Sigma_new.diag().clamp(max=sigma_max)
                Sigma_new = Sigma_new - torch.diag(Sigma_new.diag()) + torch.diag(diag)
                cma_state["Sigma"] = Sigma_new

                # Knot-space mean EMA (uses original weights: rank for cma_rank, top-10% for cma_dial)
                mu_flat = cma_state["mean"].reshape(-1)
                weighted_eps_mean = (weights[:, None] * eps).sum(dim=0)
                mu_new = mu_flat + weighted_eps_mean
                cma_state["mean"] = (
                    (1 - eta_mu) * mu_flat + eta_mu * mu_new
                ).reshape(num_knots, config.nu)

                # Full-horizon ctrls EMA
                ctrls_mean = (1 - eta_mu) * ctrls + eta_mu * ctrls_mean

                cma_state["generation"] = cma_state.get("generation", 0) + 1

        # Trace downsampling for visualization
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(
                0,
                config.num_samples - 1,
                steps=n_uni,
                dtype=torch.long,
                device=config.device,
            )
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        idx_top = (
            torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        # Build info dict
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
        info["improvement"] = float(rews_np.max() - rews_np[0])
        info["rew_u0"] = float(rews_np[0])
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()

        if "trace" in rollout_info:
            info["trace_sample"] = rollout_info["trace"][sel_idx].cpu().numpy()
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        return ctrls_mean, terminate, info

    return optimize_once


def make_optimize_fn(
    optimize_once,
):
    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
    ):
        """Full optimization loop at a given control step.

        Handles DIAL-MPC annealing, Pure MPPI, and CMA covariance adaptation
        depending on config.optimizer_mode ("dial" | "mppi" | "cma_rank").
        """
        mode = getattr(config, "optimizer_mode", "dial")

        # Initialize CMA state for full-covariance modes (cma_rank and cma_dial)
        cma_state = None
        if mode in ("cma_rank", "cma_dial"):
            from spider.optimizers.mppi_cma_full import _knots_from_ctrls

            num_knots = int(round(config.horizon / config.knot_dt))
            d = num_knots * config.nu
            mean_init = _knots_from_ctrls(ctrls, config)
            # Initialize Sigma_0 from per-dimension noise_scale (respects pos/rot/joint scaling)
            # config.noise_scale shape: (N, num_knots, nu); sample[0] is zeroed (exploit),
            # so take sample[1] which has the regular exploration scale.
            ns_flat = config.noise_scale[1].reshape(-1).to(config.device)  # (d,)
            cma_state = {
                "mean": mean_init,
                "Sigma": torch.diag(ns_flat**2).to(dtype=torch.float32),
                "generation": 0,
            }

        # Noise-annealing schedule: DIAL uses beta_traj decay; MPPI/CMA use constant 1.0
        sample_params_list = []
        for i in range(config.max_num_iterations):
            if mode == "dial":
                sample_params = {"global_noise_scale": config.beta_traj**i}
            else:
                sample_params = {"global_noise_scale": 1.0}
            sample_params_list.append(sample_params)

        infos = []
        improvement_history = []
        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once(
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

            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if (
                len(improvement_history) >= config.improvement_check_steps
                and not terminate_early_stopping
            ):
                recent_improvements = improvement_history[
                    -config.improvement_check_steps :
                ]
                if all(
                    imp < config.improvement_threshold for imp in recent_improvements
                ):
                    break

        # Temporal Gaussian smoothing — applied ONCE after all iters, CMA modes only
        smooth_window = getattr(config, "cma_ctrl_smooth_window", 0)
        if smooth_window > 1 and mode in ("cma_rank", "cma_dial"):
            H, nu = ctrls.shape
            w = smooth_window | 1  # ensure odd
            sigma = w / 4.0
            x = torch.arange(w, dtype=torch.float32, device=config.device) - w // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            c = ctrls.T.unsqueeze(0)                              # (1, nu, H)
            k = kernel.view(1, 1, w).expand(nu, 1, w).contiguous()  # (nu, 1, w)
            c_smooth = F.conv1d(c, k, padding=w // 2, groups=nu)  # (1, nu, H)
            ctrls = c_smooth.squeeze(0).T                          # (H, nu)

        # Pad to max_num_iterations with zeros so output shape is always consistent
        fake_info = {}
        for k, v in infos[0].items():
            fake_info[k] = np.zeros_like(v)
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)
        info_aggregated = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])
        return ctrls, info_aggregated

    return optimize
