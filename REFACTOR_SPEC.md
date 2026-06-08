# Optimizer Refactor: Unified sampling.py with 3 Update Modes

## Goal
Unify all optimizer logic into `spider/optimizers/sampling.py` so that DIAL-MPC, Pure MPPI, and MPPI-CMA share the same rollout infrastructure (DR, terminate-resample, per-knot noise_scale). After this, `mppi_cma_full.py` can be retired.

## Current Architecture
- `sampling.py`: rollout + DIAL-MPC update (top-10% softmax + annealing via `beta_traj`)
- `mppi_cma_full.py`: separate rollout (no DR, no terminate-resample) + CMA covariance update

## Key Differences to Reconcile

| Feature | sampling.py | mppi_cma_full.py |
|---|---|---|
| DR multi-rollout | ✅ loops env_params, takes min_rew | ❌ single env_param |
| Terminate resample | ✅ in rollout() | ✅ (uses same rollout) |
| Noise source | `config.noise_scale` (per-knot, per-DOF) × randn | Full covariance Cholesky in flat knot space |
| Weight scheme | Top-10% → normalize → softmax | Full-N softmax or rank-based |
| Mean update | Direct weighted mean (η_μ=1) | EMA: (1-η_μ)×old + η_μ×new |
| Covariance | None (isotropic, annealed via beta_traj) | Full Σ with EMA update |
| Exploit sample | `noise_scale[0] *= 0` (in get_noise_scale) | `ctrls_samples[0] = ctrls` |

## Design

### 1. New config field in `Config` (config.py)

```python
optimizer_mode: str = "dial"  # "dial" | "mppi" | "cma"
```

### 2. Sampling: `_sample_ctrls_impl` changes

Add optional `cma_state` parameter to `sample_ctrls`:

```python
def _sample_ctrls_impl(config, ctrls, sample_params=None, cma_state=None):
    global_noise_scale = sample_params.get("global_noise_scale", 1.0)
    
    if cma_state is not None and cma_state.get("Sigma") is not None:
        # CMA mode: sample from full covariance in knot space
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu
        N = config.num_samples
        Sigma = cma_state["Sigma"]
        
        # Cholesky
        Sigma = 0.5 * (Sigma + Sigma.T)
        jitter = getattr(config, "mppi_cma_jitter", 1e-4)
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(Sigma + (jitter * (10**attempt)) * torch.eye(d, device=config.device))
                break
            except torch.linalg.LinAlgError:
                if attempt == 4:
                    L = torch.diag(Sigma.diag().clamp(min=jitter).sqrt())
        
        z = torch.randn(N, d, device=config.device)
        eps_flat = (z @ L.T) * global_noise_scale  # (N, d)
        eps_knots = eps_flat.reshape(N, num_knots, config.nu)
        delta_ctrl_samples = interp(eps_knots, config.knot_steps)
        ctrls_samples = ctrls + delta_ctrl_samples
        
        # Store eps for covariance update later
        cma_state["_last_eps"] = eps_flat
        
        # Exploit sample (index 0) = unperturbed
        ctrls_samples[0] = ctrls
        cma_state["_last_eps"][0] = 0.0
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
```

Note: `torch.compile` wrapper needs updating to pass `cma_state` through.

### 3. Weight computation: `_compute_weights_impl` changes

Replace with mode-aware version:

```python
def _compute_weights_impl(rews, num_samples, temperature, mode="dial"):
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = rews[~nan_mask].min() if (~nan_mask).any() else torch.tensor(-1000.0, device=rews.device)
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
        # Pure MPPI: softmax over ALL samples, using exp(reward/λ)
        costs = -rews
        J_min = costs.min()
        w_unnorm = torch.exp(-1.0 / temperature * (costs - J_min))
        weights = w_unnorm / w_unnorm.sum()
        
    elif mode == "cma_rank":
        # CMA-ES rank-based: top-μ log-linear weights
        mu_ratio = 0.5  # could be config.cma_mu_ratio
        mu_sel = max(1, int(num_samples * mu_ratio))
        sorted_idx = torch.argsort(rews, descending=True)
        selected_idx = sorted_idx[:mu_sel]
        raw_w = torch.log(torch.tensor(mu_sel + 0.5, device=rews.device)) - \
                torch.log(torch.arange(1, mu_sel + 1, device=rews.device, dtype=torch.float32))
        weights = torch.zeros_like(rews)
        weights[selected_idx] = raw_w / raw_w.sum()
    
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    
    return weights, nan_mask
```

### 4. `optimize_once` changes

After weight computation, add CMA covariance + mean EMA update:

```python
def optimize_once(config, env, ctrls, ref_slice, env_params=[{}], sample_params=None, cma_state=None):
    mode = getattr(config, "optimizer_mode", "dial")
    
    # 1. Sample (pass cma_state for CMA mode)
    ctrls_samples = sample_ctrls(config, ctrls, sample_params, cma_state)
    ctrls_samples[0] = ctrls  # exploit sample
    
    # 2. Rollout with DR (UNCHANGED from current sampling.py)
    min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
    for env_param in env_params:
        ctrls_samples, rews, terminate, rollout_info = rollout(config, env, ctrls_samples, ref_slice, env_param)
        min_rew = torch.minimum(min_rew, rews)
    rews = min_rew
    
    # 3. Compute weights (mode-aware)
    weight_mode = {"dial": "dial", "mppi": "mppi", "cma": "cma_rank"}[mode]
    # Or use config.mppi_cma_mean_update for cma sub-modes
    weights, nan_mask = _compute_weights_impl(rews, config.num_samples, config.temperature, mode=weight_mode)
    
    # 4. Weighted mean of ctrls
    ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)
    
    # 5. CMA-specific updates (covariance + EMA mean)
    if mode == "cma" and cma_state is not None:
        eps = cma_state.get("_last_eps")  # (N, d) from sampling step
        if eps is not None:
            eta_mu = getattr(config, "mppi_cma_eta_mu", 0.5)
            eta_sigma = getattr(config, "mppi_cma_eta_sigma", 0.3)
            jitter = getattr(config, "mppi_cma_jitter", 1e-4)
            d = eps.shape[1]
            
            # Covariance update (before mean, per paper)
            sqrt_w = weights.sqrt()
            weighted_eps = sqrt_w[:, None] * eps
            Sigma_sample = weighted_eps.T @ weighted_eps
            Sigma = cma_state["Sigma"]
            Sigma_new = (1 - eta_sigma) * Sigma + eta_sigma * Sigma_sample + jitter * torch.eye(d, device=config.device)
            cma_state["Sigma"] = 0.5 * (Sigma_new + Sigma_new.T)
            
            # Mean EMA update (knot space)
            num_knots = int(round(config.horizon / config.knot_dt))
            mu_flat = cma_state["mean"].reshape(-1)
            weighted_eps_mean = (weights[:, None] * eps).sum(dim=0)
            mu_new = mu_flat + weighted_eps_mean
            cma_state["mean"] = ((1 - eta_mu) * mu_flat + eta_mu * mu_new).reshape(num_knots, config.nu)
            
            # EMA on full-horizon ctrls too
            ctrls_mean = (1 - eta_mu) * ctrls + eta_mu * ctrls_mean
            
            cma_state["generation"] = cma_state.get("generation", 0) + 1
    
    # 6. Build info dict (UNCHANGED)
    ...
    
    return ctrls_mean, terminate, info
```

### 5. `optimize` (outer loop) changes

Add CMA state initialization:

```python
def optimize(config, env, ctrls, ref_slice):
    mode = getattr(config, "optimizer_mode", "dial")
    
    # CMA state init
    cma_state = None
    if mode == "cma":
        num_knots = int(round(config.horizon / config.knot_dt))
        d = num_knots * config.nu
        from spider.optimizers.mppi_cma_full import _knots_from_ctrls
        mean_init = _knots_from_ctrls(ctrls, config)
        sigma0 = getattr(config, "cma_sigma0", 0.3)
        cma_state = {
            "mean": mean_init,
            "Sigma": (sigma0 ** 2) * torch.eye(d, device=config.device),
            "generation": 0,
        }
    
    # Annealing schedule
    sample_params_list = []
    for i in range(config.max_num_iterations):
        if mode == "dial":
            sample_params = {"global_noise_scale": config.beta_traj ** i}
        else:
            # Pure MPPI and CMA: no annealing (global_noise_scale = 1.0)
            sample_params = {"global_noise_scale": 1.0}
        sample_params_list.append(sample_params)
    
    # Optimization loop (UNCHANGED structure)
    for i in range(config.max_num_iterations):
        ctrls, terminate, info = optimize_once(
            config, env, ctrls, ref_slice,
            config.env_params_list[i],
            sample_params_list[i],
            cma_state,  # None for dial/mppi
        )
        ...  # early stopping logic unchanged
```

### 6. Config changes (config.py)

Add one field:
```python
optimizer_mode: str = "dial"  # "dial" | "mppi" | "cma"
```

### 7. Caller changes

Wherever `mppi_cma_full` is imported/used (e.g., `run_closed_loop_compare.py`, any experiment scripts), switch to using `sampling.py` with `config.optimizer_mode = "cma"`.

### 8. What NOT to change
- `rollout()` function: completely unchanged
- `make_rollout_fn()`: completely unchanged  
- Early stopping logic: unchanged
- Info dict construction: unchanged
- Trace downsampling: unchanged

### 9. After verification
- `mppi_cma_full.py` can be deprecated/removed
- Keep `_knots_from_ctrls` utility (move to utils or keep as import)

## Testing
1. Run with `optimizer_mode="dial"` → should produce identical results to current sampling.py
2. Run with `optimizer_mode="mppi"` → pure MPPI (no annealing, full-N softmax)
3. Run with `optimizer_mode="cma"` → MPPI-CMA with full covariance
4. Compare all three on the same task/seed to verify correctness
