# 05 — SPIDER 架構分析

## 1. 專案總覽

SPIDER 是 Meta FAIR 開發的 trajectory generator，目標是為 dexterous manipulation 生成物理可行的機械手軌跡。

**問題設定：** 給定一段人手操作物體的 kinematic reference trajectory（從 MANO hand model 擷取），SPIDER 在 MuJoCo 物理引擎中透過 GPU-batched sampling-based optimization，重新生成一條滿足物理約束的軌跡。換句話說——人手怎麼抓的，機械手就怎麼跟著抓，但要符合物理定律。

**核心方法：** Model Predictive Path Integral (MPPI) control。這是一種 sampling-based online planning 方法：

- 大量 sample 控制序列（GPU 上平行跑數千條）
- 在模擬器中 rollout 每條序列，算 cost
- 用 exponential weighting 把好的序列加權平均，得到下一步的最佳控制
- Receding horizon：每個 tick 前移窗口，重複上述流程

關鍵特性：**不需要訓練任何 neural network weights**。所有計算在 inference time 完成，靠的是暴力 GPU 平行模擬 + 統計最優化。

## 2. Entry Point 與執行流程

### 2.1 從 CLI 到 main()

```
spider-mjwp  ←  setup.py console_scripts 定義
    │
    ▼
examples/run_mjwp.py : main()
    │  使用 @hydra.main 裝飾器
    ▼
run_main(config)  →  parse Hydra config  →  main(config)
```

Hydra 負責 config management：所有超參數（horizon length、sample 數量、noise scale 等）都透過 YAML config 注入，CLI 可以 override。

### 2.2 main() 完整執行流程

```
main(config)
  │
  ├─ (1) process_config(config)
  │      計算 derived fields：總步數、noise_scale、model dimensions
  │
  ├─ (2) load_data(config, data_path)
  │      讀取 kinematic reference .npz
  │      插值到 sim_dt（模擬時間步長）
  │
  ├─ (3) setup_env(config, ref_data)
  │      建立 MJWPEnv：
  │        - CPU-side MuJoCo model（單一 world，用於狀態管理）
  │        - GPU-side MuJoCo Warp batched data（數千個平行 world）
  │        - CUDA graph capture（加速重複計算）
  │
  ├─ (4) setup_mj_model(config)
  │      另建一個 CPU-side MuJoCo model，專給 viewer 渲染用
  │
  ├─ (5) 建立 optimizer chain
  │      make_rollout_fn()          → 定義單次 rollout 邏輯
  │        → make_optimize_once_fn() → 包裝成單次 MPPI 優化
  │          → make_optimize_fn()    → 包裝成完整多輪優化
  │
  ├─ (6) 設定 domain randomization + contact guidance
  │      env_params_list：隨機化物理參數（摩擦、質量等）
  │      contact guidance gain schedule：接觸引導力的增減排程
  │
  ├─ (7) ═══ 主迴圈（每個 control tick）═══
  │      │
  │      ├─ optimize(config, env, ctrls, ref_slice)
  │      │    完整 MPPI 優化：sample → rollout → weight → update
  │      │
  │      ├─ step_env() × ctrl_steps
  │      │    用優化後的控制信號執行物理模擬
  │      │
  │      ├─ sync_env()
  │      │    同步 world 0（真實世界）的 state 到所有平行 worlds
  │      │
  │      └─ Receding horizon shift
  │           丟棄已執行的控制序列前段
  │           尾部補上 reference trajectory 的下一段
  │
  └─ (8) 儲存結果
         trajectory_mjwp.npz（軌跡數據）
         video（渲染影片）
```

整個流程是一個標準的 MPC loop：優化 → 執行 → 觀測 → 移動窗口 → 重複。

## 3. 模組分類與職責

SPIDER 的模組可以分成三層：**Core Pipeline**（執行時必須）、**Preprocessing**（離線前處理）、**Utilities**（輔助工具）。

### 3.1 Core Pipeline

這六個模組構成了 runtime 的完整執行鏈。

| 模組 | 檔案 | 規模 | 職責 |
|------|------|------|------|
| **Config** | `spider/config.py` | ~380 行 | Config dataclass 定義所有超參數；`process_config()` 計算 derived fields（步數、維度、noise schedule） |
| **I/O** | `spider/io.py` | ~154 行 | 載入 kinematic reference `.npz`、時間軸插值、路徑解析工具 |
| **Interpolation** | `spider/interp.py` | ~141 行 | 0th / 1st / 2nd order hold 插值，將 knot points 展開到完整 horizon |
| **Simulator** | `spider/simulators/mjwp.py` | ~1237 行 | 最大的模組。MuJoCo Warp batched simulation：環境初始化、GPU step、reward 計算、state 同步 |
| **Optimizer** | `spider/optimizers/sampling.py` | ~300 行 | MPPI 核心：noise sampling → parallel rollout → exponential weighting → control update |
| **Entry Point** | `examples/run_mjwp.py` | ~380 行 | Hydra CLI 介面 + 主控迴圈，串接上述所有模組 |

**資料流向：**

```
Config ──→ I/O（載入 reference）──→ Interpolation（插值到 sim_dt）
  │                                        │
  ▼                                        ▼
Entry Point ←── Optimizer ←──────── Simulator
  （主迴圈）      （MPPI 優化）      （GPU batched rollout + reward）
```

### 3.2 Preprocessing

離線執行、只跑一次的前處理模組。負責把人手資料轉換成機械手可用的格式。

| 模組 | 檔案 | 規模 | 職責 |
|------|------|------|------|
| **IK Retargeting** | `spider/preprocess/ik.py` | ~809 行 | MANO hand keypoints → robot joint angles。使用 MuJoCo IK solver with mocap constraints 做 retargeting |
| **Contact Detection** | `spider/preprocess/detect_contact.py` | ~459 行 | 回放 MANO 軌跡，偵測 fingertip 與 object 之間的接觸事件，生成 contact schedule |
| **Mesh Decompose** | `spider/preprocess/decompose.py` | ~113 行 | 使用 CoACD 做 convex decomposition，把複雜 mesh 拆成凸包集合供碰撞檢測使用 |

**Preprocessing 的輸出就是 Core Pipeline 的輸入：**
- IK Retargeting → kinematic reference `.npz`（餵給 `load_data()`）
- Contact Detection → contact schedule（餵給 contact guidance）
- Mesh Decompose → collision geometry（嵌入 MuJoCo model XML）

### 3.3 Utilities

輔助性模組，不影響核心邏輯。

| 模組 | 檔案 | 職責 |
|------|------|------|
| **MuJoCo Utils** | `spider/mujoco_utils.py` | Viewer helper 函數（~35 行，薄封裝） |
| **Viewers** | `spider/viewers/` | 多種渲染後端：Rerun、Viser、MuJoCo native viewer |

<!-- PART 2 CONTINUES BELOW -->
