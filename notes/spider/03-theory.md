# 03 — 理論深入

## 1. MPPI 採樣邏輯

### 採樣單位
每輪 optimization iteration 對當前 control trajectory `ctrls (horizon_steps, nu)` 加 noise，產生 1024 條完整 horizon 長度軌跡，不是單步採樣。

### Noise 生成（Knot Point Interpolation）
不是每個 timestep 獨立加 noise。在少數 knot points 上採樣 Gaussian noise，用 `interp()` 插值到整個 horizon，做時間平滑。
- Code: `_sample_ctrls_impl` → `knot_samples = torch.randn_like(config.noise_scale) * config.noise_scale * global_noise_scale` → `delta_ctrl_samples = interp(knot_samples, config.knot_steps)`

### 退火機制
`global_noise_scale = config.beta_traj ** i`，`beta_traj < 1`，跨 iteration 指數衰減（不是跨 timestep）。

### 權重計算（`_compute_weights_impl`）
- Hard cutoff: top 10% samples (`top_k = max(1, int(0.1 * num_samples))`)
- Normalize: `(top_rews - mean) / (std + 1e-2)`
- Softmax: `softmax(normalized / temperature)` — temperature 控制銳度，跟 10% cutoff 獨立
- 最終: `ctrls_mean = (weights * ctrls_samples).sum(dim=0)`

## 2. Receding Horizon 實作

- 執行前 `ctrl_steps` 步（不是只有 u₀）
- 砍掉已執行部分，尾巴補 reference trajectory control（warm start）
- 同步狀態後重新 optimize — "Plan long, act short, shift, repeat"

## 3. Contact-Guided Gain Scheduling（Homotopy）

- 物體 virtual actuators 初始 Kp/Kd gains 大，引導物體沿 reference
- 每輪 iteration 乘 `guidance_decay_ratio` 衰減
- 最後一輪強制 gains = 0，純 contact
- 本質是 homotopy: f_easy（平滑）→ f_real（不連續）
- 必要性：零 gains 時 optimizer 要同時解手指靠近 + 接觸力 + 軌跡追蹤，搜索空間太大

## 4. 理論限制

### 限制 1：最優性無保證
- Softmax weighted average 無收斂到 local minimum 的保證
- 多條 feasible trajectory 加權平均不一定 feasible（非凸 feasible set 上 convex combination 不封閉）
- 無 actuator limit projection — MuJoCo 會 clamp 但 optimizer 不感知

### 限制 2：解空間的結構性問題
- Feasible set S_t 隨任務推進收縮（contact transition 附近特別劇烈）
- Control → outcome 映射在 contact boundary 不連續（非 Lipschitz）
- |S_t △ S_{t+1}| 小 ≠> |f(S_t) △ f(S_{t+1})| 小
- 不可行三層結構：控制可行（actuator limits）⊆ 數學可行（homotopy path 存在性）⊆ 物理可行（物體不掉）

### 限制 3：泛化性問題
- Reference 來自人手 retarget，mapping g: C_human → C_robot 不保證 surjective，q_ref 可能不在 robot reachable set
- Contact 作為隱式 constraint，optimizer 對 contact mode 結構盲
- Contact modes 隨 DOF 指數增長 O(2^{nm})，1024 samples 在高維空間覆蓋率指數衰減
- 新 robot morphology 可能導致 homotopy path 在 α→1 時斷裂（robot 幾何不支持所需 contact wrench）

## 5. SPIDER 的設計如何對抗這些限制

| 問題 | 對策 | 機制 |
|---|---|---|
| 映射不連續 | Gain scheduling | 軟化 f_real 為 f_easy，逐步過渡 |
| Sample 碎片化 | Terminate resampling | 好 sample 覆蓋失敗 sample（particle filter） |
| 高維搜索 | Knot point interpolation | 限制搜索在時間平滑子流形上 |

---

## Phase 3 完成條件

- [x] 能用自己的話解釋核心演算法的每一步
- [x] 列出 ≤3 個理論上的限制或假設
