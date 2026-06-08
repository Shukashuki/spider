# 01 — Minimal Run + 知識對接

## Demo 結果摘要

| 項目 | 值 |
|---|---|
| Task | `p36-tea` |
| Dataset | `gigahand` |
| Robot | `xhand` |
| Embodiment | `bimanual` (雙手) |
| Samples | 1024 |
| Total time | 634.3s (~10.5 min) |
| sim_steps | 680/676 |
| Realtime rate | ~0.01–0.05x |
| Object tracking error | pos=0.0643, quat=0.6682 |
| Output | `example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/trajectory_mjwp.npz` |
| 環境 | WSL2, RTX 3060 Ti 8GB, CUDA 12.8, `show_viewer=false save_video=false` |

---

## 核心概念表

| # | 概念 | 在 SPIDER 中的角色 | 對應已知知識 | 狀態 |
|---|---|---|---|---|
| 1 | **Sampling-Based Trajectory Optimization (MPPI/DIAL-MPC)** | 核心 optimizer。對當前 control trajectory 加 noise 產生 1024 個 sample，parallel rollout 後取 top 10% 做 softmax weighted average 得到更新的 control。 | Aero740 (MPC shooting methods) + ROB422 (sampling-based planning) | 模糊 |
| 2 | **Receding Horizon Control (MPC)** | 主迴圈結構。每一步在 `horizon_steps` 上做 optimization，只執行前 `ctrl_steps` 步，然後把 horizon 往前推，重複。 | Aero740 (MPC receding horizon) | 已知 |
| 3 | **GPU-Parallel Physics Simulation (MuJoCo Warp)** | 把 MuJoCo 編譯成 NVIDIA Warp kernels，1024 個 environment 在 GPU 上並行模擬。這是讓 sampling-based 方法在合理時間內跑完的關鍵。 | 知道 MuJoCo，但 Warp 並行化是新的 | 需學習 |
| 4 | **Motion Retargeting Cost Formulation** | Reward = -‖Δqpos‖₂ (weighted) - λ_vel‖Δqvel‖₂ - λ_contact‖Δcontact_pos‖₂。追蹤人類動作的 reference trajectory，讓機器人手盡量模仿。 | Aero740 (quadratic cost / tracking cost) + ROB422 (grasp metrics) | 模糊 |
| 5 | **Contact-Guided Optimization with Gain Scheduling** | 在 object actuator 上設 Kp/Kd gains 引導物體跟隨 reference，然後逐 iteration 衰減 gains 直到最後一步歸零——讓 robot 自己「接住」物體，不靠 actuator 作弊。 | Aero740 (constraint relaxation / homotopy) + ME564 (gain scheduling) | 需學習 |

---

## 知識連結點

### 1. Sampling-Based Trajectory Optimization → Aero740 + ROB422

**你已經知道的：**
- Aero740: MPC 用 shooting method 把 control sequence 往前 rollout，算 cost，然後 gradient descent 更新 control
- ROB422: sampling-based motion planning（RRT / PRM）——在 configuration space 隨機 sample，找好的路徑

**SPIDER 的做法（差在哪）：**
- 不算 gradient。完全靠 sampling + weighted averaging（類似 CEM / MPPI）
- 對 reference control 加 Gaussian noise → 產生 N=1024 個 candidate trajectories
- 每個 candidate 在 physics simulator 裡 rollout → 算 reward
- **只取 top 10%**，做 softmax weighted average 得到新的 control（`_compute_weights_impl`）
- 這跟 Aero740 的 gradient-based shooting 最大差別：**不需要 dynamics model 的 gradient**，只需要 forward simulation
- 跟 ROB422 的 sampling-based planning 的差別：RRT samples in config space seeking feasibility，MPPI samples in control space seeking optimality

**需要搞懂的：**
- MPPI (Model Predictive Path Integral) 的理論基礎——為什麼 softmax weighted average 會收斂到好的 control？（跟 information-theoretic optimal control 有關）
- `temperature` 參數的作用——控制 exploitation vs exploration

### 2. GPU-Parallel Simulation → MuJoCo + CUDA

**你已經知道的：**
- MuJoCo 的 API（`mj_step`, `mj_kinematics` 等）
- CUDA programming 的基本概念
- PyTorch 的 GPU tensor 操作

**SPIDER 的做法：**
- MuJoCo Warp 把 MuJoCo 的 physics engine 編譯成 Warp kernels，跑在 GPU 上
- 一次模擬 1024 個 environment（`num_samples` 個 world），每個有不同的 control input
- 用 `wp.to_torch()` 在 Warp tensors 和 PyTorch tensors 之間轉換
- 這是讓 sampling-based MPC 實際可行的瓶頸突破——CPU 上 1024 次 rollout 太慢

**需要搞懂的：**
- Warp 的 kernel 編譯模型（為什麼第一次跑要 compile 那麼久？）
- `save_state` / `load_state` 的機制——怎麼在 GPU 上做 environment checkpoint 和 reset

### 3. Contact-Guided Optimization → Aero740 homotopy + ME564 gain scheduling

**你已經知道的：**
- Aero740: constraint relaxation（先放寬 constraints，逐步收緊）
- ME564: PD controller 的 gain tuning

**SPIDER 的做法：**
- 物體（tea cup 等）有 virtual actuators，初始時用 Kp/Kd gains 讓物體「黏著」reference trajectory
- 每個 optimization iteration 把 gains 乘以 `guidance_decay_ratio`（衰減）
- 最後一個 iteration 強制 gains = 0 → 物體完全靠 robot 手指的接觸力控制
- 這本質上是一種 **curriculum / homotopy 方法**：先解一個簡單問題（物體有 actuator 幫忙），再逐步過渡到真實問題（純 contact）

**需要搞懂的：**
- 為什麼不一開始就零 gains？（因為 sampling optimizer 很難從零開始找到正確的 contact 模式）
- `guidance_decay_ratio` 的選擇——衰減太快會失敗，太慢浪費時間

---

## 下一步問題（Phase 2 需要深挖）

1. **環境可復現性**：把今天所有的修復步驟（git-lfs, python3.12-dev, show_viewer=false）整理成可一鍵復現的腳本
2. **其他 dataset/robot 組合**：oakink dataset、humanoid 等能不能跑？是否需要不同的 config？
3. **Object tracking error 的意義**：pos=0.0643 是好還是差？需要跟論文的 baseline 比較
4. **Realtime rate 改善**：0.01x realtime 能不能透過 tuning `num_samples`、`horizon_steps` 改善？trade-off 是什麼？
5. **Warp kernel cache**：第二次跑同樣的 task 會快多少？cache 機制是什麼？
