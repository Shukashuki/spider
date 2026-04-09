"""CMA-ES optimizer for SPIDER trajectory optimization.

Simplified diagonal-covariance CMA-ES operating in the knot-point space.
Uses rank-based weighted recombination for mean and sigma updates,
then interpolates to the full horizon for rollout evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from spider.config import Config
from spider.interp import interp


def make_rollout_fn_cma(*args):
    """Reuse the same rollout function factory as MPPI."""
    from spider.optimizers.sampling import make_rollout_fn

    return make_rollout_fn(*args)


def _compute_recombination_weights(mu: int, device: torch.device) -> torch.Tensor:
    """Compute CMA-ES log-based recombination weights for top-mu individuals.

    w_i = log(mu + 0.5) - log(i + 1),  i = 0..mu-1, then normalised.
    """
    raw = torch.log(torch.tensor(mu + 0.5, device=device)) - torch.log(
        torch.arange(1, mu + 1, dtype=torch.float32, device=device)
    )
    return raw / raw.sum()


def _knots_from_ctrls(ctrls: torch.Tensor, config: Config) -> torch.Tensor:
    """Extract knot-point values from a full-horizon ctrl tensor.

    ctrls: (horizon_steps, nu)
    returns: (num_knots, nu)  by slicing every knot_steps.
    """
    num_knots = int(round(config.horizon / config.knot_dt))
    indices = torch.arange(num_knots, device=ctrls.device) * config.knot_steps
    indices = indices.clamp(max=ctrls.shape[0] - 1).long()
    return ctrls[indices]


def make_optimize_once_fn_cma(rollout):
    """Return a single-iteration CMA-ES optimiser step."""

    def optimize_once_cma(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
        env_params=None,
        sample_params=None,
        cma_state=None,
    ):
        """One generation of CMA-ES.

        Args:
            config: Config object.
            env: Environment handle (passed through to rollout).
            ctrls: Current best controls, shape (horizon_steps, nu).
            ref_slice: Reference trajectory slice.
            env_params: List of env-parameter dicts for domain randomisation.
            sample_params: Dict; may contain ``global_noise_scale``.
            cma_state: Mutable dict carrying CMA state across generations.
                       Keys: mean (num_knots, nu), sigma (num_knots, nu), generation (int).

        Returns:
            ctrls_mean: Updated controls (horizon_steps, nu).
            terminate: Termination flag tensor from rollout.
            info: Dict with reward statistics (same keys as MPPI).
        """
        if env_params is None:
            env_params = [{}]
        if sample_params is None:
            sample_params = {}

        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))
        lam = config.num_samples  # population size
        mu = max(1, int(lam * config.cma_mu_ratio))  # parents
        global_noise_scale = sample_params.get("global_noise_scale", 1.0)

        # ---- initialise CMA state if needed --------------------------------
        if cma_state is None:
            raise ValueError("cma_state must be provided (create it in optimize_cma)")
        mean = cma_state["mean"]  # (num_knots, nu)
        sigma = cma_state["sigma"]  # (num_knots, nu)

        # ---- 1. sample population ------------------------------------------
        # z ~ N(0, I), then x = mean + sigma * z * global_noise_scale
        z = torch.randn(lam, num_knots, config.nu, device=device)
        knot_samples = mean.unsqueeze(0) + sigma.unsqueeze(0) * z * global_noise_scale
        # keep first sample as the mean (exploit sample)
        knot_samples[0] = mean

        # ---- 2. interpolate to full horizon ---------------------------------
        ctrls_samples = interp(knot_samples, config.knot_steps)  # (lam, horizon_steps, nu)

        # ---- 3. rollout -----------------------------------------------------
        ctrls_samples, rews, terminate, rollout_info = rollout(
            config, env, ctrls_samples, ref_slice, env_params[0] if env_params else {}
        )

        # ---- 4. rank & select top-mu ----------------------------------------
        sorted_idx = torch.argsort(rews, descending=True)
        elite_idx = sorted_idx[:mu]

        weights = _compute_recombination_weights(mu, device)  # (mu,)

        # elite knot-point values (from pre-interp space)
        elite_knots = knot_samples[elite_idx]  # (mu, num_knots, nu)

        # ---- 5. update mean -------------------------------------------------
        mean_old = mean.clone()
        mean_new = (weights[:, None, None] * elite_knots).sum(dim=0)  # (num_knots, nu)

        # ---- 6. update sigma (diagonal) -------------------------------------
        diff = elite_knots - mean_old.unsqueeze(0)  # (mu, num_knots, nu)
        sigma_new = torch.sqrt(
            (weights[:, None, None] * diff.pow(2)).sum(dim=0).clamp(min=1e-10)
        )  # (num_knots, nu)

        # ---- persist state --------------------------------------------------
        cma_state["mean"] = mean_new
        cma_state["sigma"] = sigma_new
        cma_state["generation"] = cma_state.get("generation", 0) + 1

        # ---- 7. reconstruct best ctrls from updated mean --------------------
        ctrls_mean = interp(mean_new.unsqueeze(0), config.knot_steps).squeeze(0)

        # ---- 8. build info dict (same keys as MPPI) -------------------------
        rews_np = rews.cpu().numpy()
        info: dict = {}

        # per-metric stats from rollout_info (mirrors MPPI)
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

        # trace visualisation (same down-sampling logic as MPPI)
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

    return optimize_once_cma


def make_optimize_fn_cma(optimize_once_cma):
    """Return the full CMA-ES optimisation loop (mirrors MPPI's make_optimize_fn)."""

    def optimize_cma(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice,
    ):
        """Run CMA-ES for up to ``max_num_iterations`` generations.

        Returns:
            ctrls: Optimised controls (horizon_steps, nu).
            info_aggregated: Dict of arrays stacked over iterations.
        """
        device = config.device
        num_knots = int(round(config.horizon / config.knot_dt))

        # ---- initialise CMA state from current ctrls -----------------------
        mean_init = _knots_from_ctrls(ctrls, config)  # (num_knots, nu)
        sigma_init = torch.full(
            (num_knots, config.nu),
            config.cma_sigma0,
            device=device,
            dtype=torch.float32,
        )
        cma_state = {
            "mean": mean_init,
            "sigma": sigma_init,
            "generation": 0,
        }

        # noise-annealing schedule (same as MPPI)
        sample_params_list = []
        for i in range(config.max_num_iterations):
            sample_params_list.append({"global_noise_scale": config.beta_traj ** i})

        infos = []
        improvement_history: list[float] = []

        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once_cma(
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

            # early stopping (same logic as MPPI)
            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if (
                len(improvement_history) >= config.improvement_check_steps
                and not terminate_early_stopping
            ):
                recent = improvement_history[-config.improvement_check_steps :]
                if all(imp < config.improvement_threshold for imp in recent):
                    break

        # pad infos to max_num_iterations with zeros (mirrors MPPI)
        fake_info = {k: np.zeros_like(v) for k, v in infos[0].items()}
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)

        info_aggregated: dict = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])

        return ctrls, info_aggregated

    return optimize_cma
