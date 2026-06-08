## 6. 可重用模組分析

SPIDER 的 codebase 中有幾個模組具備足夠的獨立性，可以直接抽出來用在其他專案裡。以下挑三個最值得關注的。

### 6.1 `spider/interp.py` — 通用時序插值器

這個模組完全獨立，唯一的外部依賴是 PyTorch。它把時序插值封裝成一個乾淨的函數介面：

```
interp(src: (N, H, D), n: int, order: int) -> (N, H*n, D)
```

支援三種插值模式：
- **0th order hold**：零階保持，每個 knot 值直接複製 n 次
- **1st order hold**：線性插值
- **2nd order hold**：二次插值，基於 `F.interpolate` 實現

底層走的是 PyTorch 的 `F.interpolate`，所以天然支援 GPU batch 運算。適用場景很廣——signal processing、trajectory smoothing、把 knot-based 的稀疏表示展開成 dense signal，基本上任何需要在 GPU 上做 batch 時序插值的地方都能直接拿來用。

另外搭配同模組的 `get_slice()` 函數，可以從插值後的 dense signal 中做 sliding window extraction，這在 MPC 的 receding horizon 場景裡很實用。

**可移植性：⭐⭐⭐⭐⭐** — 零耦合，拷貝單一檔案即可使用。

### 6.2 `spider/optimizers/sampling.py` — 通用 MPPI 框架

這是整個 SPIDER 裡設計最精巧的模組。它用純 functional programming 風格實現了一套完整的 MPPI（Model Predictive Path Integral）optimizer，透過層層 closure 組裝：

- `make_rollout_fn()` — 接收 `step_env`、`get_reward`、`save_state`、`load_state` 等函數，回傳一個 `rollout()` closure
- `make_optimize_once_fn()` — 在 rollout 之上加入 noise sampling 和 weighted averaging，回傳單步優化 closure
- `make_optimize_fn()` — 在 optimize_once 之上加入迭代和 early stopping，回傳完整的優化流程

只要提供符合函數簽名的 `step_env`、`get_reward`、`save_state`、`load_state`，就能把這套框架套用在任何 MPC 問題上。框架自帶的功能相當豐富：

- **Terminate-resample**：偵測到 terminated trajectory 後重新 sample
- **Domain randomization**：對多組 env params 各跑一次 rollout，取 min reward（worst-case robustness）
- **Noise annealing**：跨迭代逐步降低 noise scale
- **Early stopping**：reward 收斂時提前結束

**限制**：目前 elite selection 是 hard-coded 取 top 10%，weighting 用 softmax，temperature 從 config 讀取。如果要換成其他 selection / weighting 策略，需要改原始碼。

**可移植性：⭐⭐⭐⭐** — 模組本身獨立性高，但需要使用者自行實現符合介面的 simulator 函數。

### 6.3 `spider/simulators/mjwp.py` — Batched State Management

嚴格來說不是整個 `mjwp.py`，而是其中的 state management 函數群：

- `save_state()` — 將當前所有 sample 的 simulation state 存成 checkpoint
- `load_state()` — 從 checkpoint 恢復 state
- `sync_env()` — 同步環境狀態
- `copy_sample_state()` — 在 sample 之間複製 state

這組函數解決了 simulation-based planning 最核心的問題：「rollout 完之後怎麼 reset 回去」。在 MPPI 這類 sampling-based optimizer 中，每輪需要從同一個起點 rollout 數百條 trajectory，然後回到原點重來。這組函數提供了高效的 batched checkpoint / restore 機制。

**限制**：與 MuJoCo Warp 的資料結構高度耦合（直接操作 `wp.sim.State` 等內部物件）。移植到其他 simulator 需要重新 adapt 介面，但設計思路（雙 buffer swap、batched state copy）是通用的。

**可移植性：⭐⭐⭐** — 思路可借鑑，但程式碼本身需要大幅修改才能用在非 MuJoCo Warp 環境。


## 7. 設計模式觀察

### 7.1 Functional Closure 組裝模式

SPIDER 的 optimizer 層完全採用 functional style，這是它最有特色的架構決策。

核心思路：`make_rollout_fn()` 接收一組函數（`step_env`、`get_reward`、`save_state`、`load_state` 等），在內部組裝邏輯後回傳一個 `rollout()` closure。上層的 `make_optimize_once_fn()` 和 `make_optimize_fn()` 依樣畫葫蘆，層層包裝。

這本質上是 **Strategy Pattern 的 functional 變體**。傳統 OOP 做法會定義一個 `Simulator` 抽象類別，然後讓 `MuJoCoWarpSimulator` 去繼承。SPIDER 的做法直接跳過了繼承體系——simulator 只需要提供幾個符合簽名的函數就好。

實際效果：在 `run_mjwp.py` 裡，只要把 `mjwp` 模組的 `step`、`reward`、`save_state`、`load_state` 傳進去，一行就組裝出完整的 optimizer pipeline。要換 simulator，換一組函數就行。

### 7.2 Dataclass-as-Config 模式

`Config` 是一個包含約 60 個欄位的巨型 dataclass，透過 Hydra 從 YAML 載入。但它實際上混合了三種不同性質的資料：

| 類別 | 範例欄位 |
|------|---------|
| User-facing 超參數 | `num_samples`, `temperature`, `horizon` |
| Derived fields（由其他欄位計算得出） | `horizon_steps`, `noise_scale` |
| Runtime state（執行期間才產生） | `env_params_list`, `viewer_body_entity_and_ids` |

`process_config()` 負責在初始化時 hydrate 那些 derived fields（例如從 `horizon` 和 `dt` 算出 `horizon_steps`）。

這個設計的好處是簡單——所有東西都在一個地方，Hydra 整合也很順暢。壞處是它違反了 Single Responsibility Principle：Config 同時扮演了「使用者配置」「計算中間結果」和「runtime 狀態容器」三個角色。理想情況下應該拆成 `SimConfig`、`OptConfig`、`IOConfig` 等獨立的 dataclass。

### 7.3 GPU Batching 策略

SPIDER 的 GPU batching 採用了一個精巧的設計：所有 N 個 sample（平行 trajectory）共享同一個 MuJoCo Warp model（物理參數、mesh 等），但各自維護獨立的 simulation state（位置、速度、接觸力等）。

效能關鍵在 CUDA graph capture。`_compile_step()` 在第一次 `step()` 呼叫時錄製整個 CUDA graph（包含所有 kernel launch），後續的 step 直接 replay 這個 graph，避免了重複的 kernel launch overhead。這對於需要跑數百步 simulation 的 MPPI 來說是巨大的加速。

State management 則靠 `data_wp` / `data_wp_prev` 雙 buffer swap 來實現——一個 buffer 存當前 state，一個存前一步的 state，每步交換。這避免了每步都要分配新記憶體的開銷。

### 7.4 Domain Randomization 實作

SPIDER 的 domain randomization 採用了一個不太常見的策略：不是在 setup 階段一次性注入隨機化參數，而是在每輪 `optimize_once` 中動態切換。

流程：
1. 準備多組 `env_params`（例如不同的摩擦係數、質量等）
2. 每輪優化時，對每組 params 各跑一次完整的 rollout
3. 取所有 params 下的 **min reward** 作為最終 reward

這實現了 worst-case robustness：optimizer 會找到在「最差情況」下仍然表現良好的 trajectory。`save_env_params()` / `load_env_params()` 負責保存和恢復 active parameter group pointer，讓切換過程不需要重建 simulator。

### 7.5 程式碼品質觀察

作者是 Chaoyi Pan（Meta FAIR），開發時間大約在 2025 年 7 月到 11 月之間。

**做得好的部分：**
- 有清楚的 docstring 和 type hints
- Functional 架構設計乾淨，模組邊界清晰
- MPPI 框架的可組合性設計得很好

**可以改善的部分：**
- 大量 commented-out code 散佈在 `detect_contact.py`、`ik.py` 等檔案中
- Config dataclass 過於龐大，應拆分成多個專責的 config class
- `ik.py` 中硬編碼了不少 magic numbers（`solimp`、`solref` 的預設值等），缺乏說明
- 部分函數過長，`main()` 約 300 行，應拆分

**整體評價：** 這是典型的 research-grade code——功能完整、思路清晰、能跑出結果，但還不到 production-ready 的程度。對於學術研究和概念驗證來說完全夠用，但如果要部署到正式環境，需要做相當程度的重構和清理。
