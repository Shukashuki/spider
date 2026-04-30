"""MPPI with Covariance Matrix Adaptation (MPPI-CMA).

Delta parameterization: mean represents accumulated delta from ctrls_orig.
ctrls_orig is fixed for all iterations within a control step.
ctrls_samples = ctrls_orig + interp(mean + eps) — same structure as sampling.py.

Index 0: u0 reference = ctrls_orig (zero delta), excluded from MPPI update.
Index 1: exploit = ctrls_orig + interp(mean), eps=0.
Indices 2..N: random perturbations.
"""

from __future__ import annotations

import numpy as np
import torch

from spider.config import Config
from spider.interp import interp


def make_optimize_once_fn_mppi_cma(rollout):

    def optimize_once_mppi_cma(
        config: Config,
        env,
        ctrls_orig: torch.Tensor,
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
        global_noise_scale = sample_params.get("global_noise_scale", 1.0)

        mean_delta = cma_state["mean"]   # (num_knots, nu) delta from ctrls_orig
        sigma = cma_state["sigma"]       # (d,)
        temperature = config.temperature
        eta_mu    = config.mppi_cma_eta_mu
        eta_sigma = config.mppi_cma_eta_sigma
        jitter    = config.mppi_cma_jitter

        mu_flat = mean_delta.reshape(-1)  # (d,)

        # ---- 1. Sample delta perturbations ----
        eps = torch.randn(lam, d, device=device) * sigma.unsqueeze(0) * global_noise_scale
        u = mu_flat.unsqueeze(0) + eps   # (N, d) delta candidates
        eps[1] = 0.0
        u[1]   = mu_flat                 # exploit: current best delta

        delta_full = interp(u.reshape(lam, num_knots, config.nu), config.knot_steps)
        ctrls_samples = ctrls_orig.unsqueeze(0) + delta_full   # (N, H, nu)
        ctrls_samples[0] = ctrls_orig    # index 0: exact u0, zero delta

        # ---- 2. Rollout ----
        ctrls_samples, rews, terminate, rollout_info = rollout(
            config, env, ctrls_samples, ref_slice, env_params[0] if env_params else {}
        )

        # ---- 3. MPPI weights on indices 1..N ----
        costs = -rews[1:]
        J_min = costs.min()
        w_unnorm = torch.exp(-1.0 / temperature * (costs - J_min))
        w = w_unnorm / w_unnorm.sum()

        # ---- 4. Covariance update ----
        weighted_eps_sq = (w[:, None] * eps[1:].pow(2)).sum(dim=0)
        sigma_sq_new = (1 - eta_sigma) * sigma.pow(2) + eta_sigma * weighted_eps_sq + jitter
        sigma_new = sigma_sq_new.sqrt()

        # ---- 5. Mean delta update ----
        mu_new_flat = (1 - eta_mu) * mu_flat + eta_mu * (w[:, None] * u[1:]).sum(dim=0)

        cma_state["mean"]       = mu_new_flat.reshape(num_knots, config.nu)
        cma_state["sigma"]      = sigma_new
        cma_state["generation"] = cma_state.get("generation", 0) + 1

        # best ctrls = ctrls_orig + interp(accumulated mean delta)
        best_delta = interp(cma_state["mean"].unsqueeze(0), config.knot_steps).squeeze(0)
        ctrls_mean = ctrls_orig + best_delta

        # ---- Info ----
        rews_np = rews.cpu().numpy()
        info: dict = {}
        for k, v in rollout_info.items():
            if k in ("trace", "trace_sample"):
                continue
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if v.ndim == 1:
                info[k + "_max"]    = v.max()
                info[k + "_min"]    = v.min()
                info[k + "_median"] = np.median(v)
                info[k + "_mean"]   = v.mean()

        info["improvement"] = float(rews_np[1:].max() - rews_np[1])
        info["rew_u0"]      = float(rews_np[0])
        info["rew_max"]     = float(rews_np[1:].max())
        info["rew_min"]     = float(rews_np[1:].min())
        info["rew_median"]  = float(np.median(rews_np[1:]))
        info["rew_mean"]    = float(rews_np[1:].mean())

        if "trace" in rollout_info:
            n_uni = max(0, min(config.num_trace_uniform_samples, lam))
            n_topk = max(0, min(config.num_trace_topk_samples, lam))
            idx_uni = (
                torch.linspace(0, lam - 1, steps=n_uni, dtype=torch.long, device=device)
                if n_uni > 0 else torch.tensor([], dtype=torch.long, device=device)
            )
            idx_top = (
                torch.topk(rews, k=n_topk, largest=True).indices
                if n_topk > 0 else torch.tensor([], dtype=torch.long, device=device)
            )
            sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()
            info["trace_sample"] = rollout_info["trace"][sel_idx].cpu().numpy()
            info["trace_cost"]   = -rews[sel_idx].cpu().numpy()

        return ctrls_mean, terminate, info

    return optimize_once_mppi_cma


def make_optimize_fn_mppi_cma(optimize_once_mppi_cma):

    def optimize_mppi_cma(config: Config, env, ctrls: torch.Tensor, ref_slice):
        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu

        # fix ctrls_orig for all iterations — mean accumulates delta from it
        ctrls_orig = ctrls.clone()

        mean_init  = torch.zeros(num_knots, config.nu, device=device, dtype=torch.float32)
        sigma_init = torch.full((d,), config.cma_sigma0, device=device, dtype=torch.float32)
        cma_state  = {"mean": mean_init, "sigma": sigma_init, "generation": 0}

        sample_params_list = [
            {"global_noise_scale": config.beta_traj ** i}
            for i in range(config.max_num_iterations)
        ]

        infos: list[dict] = []
        improvement_history: list[float] = []

        for i in range(config.max_num_iterations):
            ctrls_out, terminate, info = optimize_once_mppi_cma(
                config, env, ctrls_orig, ref_slice,
                config.env_params_list[i] if hasattr(config, "env_params_list") else [{}],
                sample_params_list[i],
                cma_state,
            )
            infos.append(info)
            improvement_history.append(info["improvement"])

            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if len(improvement_history) >= config.improvement_check_steps and not terminate_early_stopping:
                recent = improvement_history[-config.improvement_check_steps:]
                if all(imp < config.improvement_threshold for imp in recent):
                    break

        fake_info = {k: np.zeros_like(v) for k, v in infos[0].items()}
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)

        info_aggregated: dict = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])

        return ctrls_out, info_aggregated

    return optimize_mppi_cma
