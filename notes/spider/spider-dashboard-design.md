# SPIDER Experiment Dashboard — Architecture Design

> **版本**: v1.0 | **環境**: WSL2 + Streamlit + RTX 3060 Ti  
> **定位**: 這份文件是實作藍圖。後續每個模組可由獨立 sub-agent 實作，互不干擾。

---

## 0. 設計原則

1. **不重造輪子** — 所有計算邏輯直接 import SPIDER 原 repo 模組；dashboard 只做 UI 膠水
2. **模組邊界清晰** — 每個 Python 檔案只做一件事，輸入輸出用 typed dataclass 或 dict schema 定義
3. **非同步優先** — 25 min per run，UI 絕不能凍結；runner 永遠跑在 background process
4. **WSL2 友好** — 無 display server，所有視覺化用 matplotlib/plotly（不依賴 Qt/OpenGL）
5. **漸進式複雜度** — 從 npz 讀檔開始，跑通基礎流程再加 runner/registry

---

## 1. 目錄結構

```
dashboard/                          # repo 根目錄，放在 spider/ 同層或 spider/dashboard/
│
├── app.py                          # Streamlit 入口，多頁面路由
├── requirements.txt                # 額外依賴（plotly, pandas, etc.）
├── config.toml                     # Streamlit server config（port, headless=true）
│
├── pages/                          # Streamlit multi-page
│   ├── 1_Overview.py               # Dashboard Overview
│   ├── 2_Single_Run.py             # Single Run Analysis
│   ├── 3_Compare.py                # Compare Experiments
│   └── 4_New_Run.py                # Run New Experiment
│
├── core/                           # 核心邏輯，與 UI 無關
│   ├── __init__.py
│   ├── registry.py                 # Experiment Registry（SQLite）
│   ├── runner.py                   # Background experiment runner
│   ├── loader.py                   # npz 載入 + 快取
│   ├── analyzer.py                 # 四大指標計算
│   └── config_builder.py           # Config dataclass 組裝
│
├── ui/                             # 可複用 UI 元件
│   ├── __init__.py
│   ├── config_panel.py             # Sidebar 四級旋鈕
│   ├── metric_cards.py             # 四大指標卡片
│   ├── charts.py                   # 所有 plotly/matplotlib 圖表
│   └── experiment_selector.py      # 實驗選擇器 widget
│
├── viz_adapters/                   # 把現有 viz 腳本包成函數
│   ├── __init__.py
│   ├── v01_reward.py               # 01: reward convergence adapters
│   ├── v02_cost.py                 # 02: cost decomposition adapters
│   ├── v03_trajectory.py           # 03: joint timeseries adapters
│   ├── v04_control.py              # 04: ctrl signal adapters
│   ├── v05_3d.py                   # 05: 3D trajectory adapters
│   ├── v06_rollout.py              # 06: cross-rollout adapters
│   └── v07_compare.py              # 07: multi-experiment adapters
│
├── data/
│   └── registry.db                 # SQLite experiment registry（gitignore）
│
└── tests/                          # 各模組單元測試
    ├── test_registry.py
    ├── test_loader.py
    └── test_analyzer.py
```

---

## 2. 模組拆分

### 2.1 `core/registry.py` — Experiment Registry

**職責**: 記錄每次實驗的 metadata、狀態、結果路徑

**資料表 schema（SQLite）**:

```
experiments
├── id          TEXT PRIMARY KEY   -- UUID
├── name        TEXT               -- 人類可讀名稱，e.g. "tea_temp0.5_s42"
├── created_at  TEXT               -- ISO8601
├── status      TEXT               -- "pending" | "running" | "done" | "failed"
├── config_json TEXT               -- 完整 Config 序列化為 JSON
├── task        TEXT               -- e.g. "p36-tea"
├── robot_type  TEXT
├── dataset_name TEXT
├── data_id     INTEGER
├── output_dir  TEXT               -- trajectory_mjwp.npz 所在目錄
├── pid         INTEGER            -- runner process PID（用於 kill）
├── duration_s  REAL               -- 實際耗時（秒）
├── error_msg   TEXT               -- 失敗時的 traceback
└── tags        TEXT               -- 逗號分隔 tag，e.g. "baseline,ablation"
```

**介面定義**:
```
create_experiment(config_json, name, tags) → experiment_id
update_status(experiment_id, status, pid=None, error_msg=None)
set_output_dir(experiment_id, output_dir)
set_duration(experiment_id, duration_s)
list_experiments(status=None, task=None, limit=50) → List[ExperimentRecord]
get_experiment(experiment_id) → ExperimentRecord
delete_experiment(experiment_id)
```

**依賴**: `sqlite3` (stdlib), `dataclasses`

---

### 2.2 `core/runner.py` — Background Runner

**職責**: 在獨立 subprocess 中執行 `examples/run_mjwp.py:main(config)`，更新 registry 狀態，串流 log

**設計**:
- 使用 `subprocess.Popen` 啟動獨立 Python process（不用 asyncio/threading，避免 Streamlit reload 問題）
- stdout/stderr 重定向到 `{output_dir}/run.log`
- 每 10 秒輪詢 `registry.db` 取得進度（Streamlit 用 `st.rerun()` 驅動輪詢）
- PID 存入 registry，支援手動 kill

**介面定義**:
```
launch_experiment(experiment_id, config, output_dir) → pid
kill_experiment(experiment_id) → bool
get_log_tail(experiment_id, lines=50) → str
estimate_progress(experiment_id) → float  # 0.0~1.0，根據 log 解析 tick 進度
```

**進度解析策略**:
- `run_mjwp.py` 每個 tick 應有 log 輸出，parse `"Tick {i}/{n_ticks}"` 格式
- 若無此格式，fallback 到 `output_dir/trajectory_mjwp.npz` 檔案大小增長曲線估算

**依賴**: `subprocess`, `pathlib`, `core/registry.py`

---

### 2.3 `core/loader.py` — Data Loader

**職責**: 載入 npz 檔案，快取，提供型別安全的資料結構

**輸入**: `output_dir: Path`  
**輸出**: `TrajectoryData` dataclass

```
TrajectoryData:
├── qpos:          np.ndarray  # (n_ticks, ctrl_steps, nq)
├── ctrl:          np.ndarray  # (n_ticks, ctrl_steps, nu)
├── rew_mean:      np.ndarray  # (n_ticks, max_iterations)
├── rew_min:       np.ndarray
├── rew_max:       np.ndarray
├── opt_steps:     np.ndarray  # (n_ticks, 1)
├── trace_cost:    np.ndarray  # (n_ticks, max_iterations, n_cost_components)
├── qpos_dist_mean: np.ndarray
├── qpos_dist_min:  np.ndarray
├── qpos_dist_max:  np.ndarray
├── trace_ref:     np.ndarray  # (n_ticks, 1, 1, horizon_steps, n_sites, 3)
└── meta:          dict        # n_ticks, ctrl_steps, nq, nu, n_cost_components
```

**介面定義**:
```
load_trajectory(output_dir: Path) → TrajectoryData
load_trajectory_cached(output_dir: Path) → TrajectoryData  # st.cache_data
invalidate_cache(output_dir: Path)
```

**依賴**: `numpy`, `spider.io` (load_data, get_processed_data_dir)

---

### 2.4 `core/analyzer.py` — Metrics Analyzer

**職責**: 從 `TrajectoryData` 計算四大指標，輸出可直接顯示的數字

**輸入**: `TrajectoryData`  
**輸出**: `ExperimentMetrics` dataclass

```
ExperimentMetrics:
├── opt_steps_mean:     float   # 平均收斂 steps（越小越好）
├── opt_steps_std:      float
├── opt_steps_per_tick: np.ndarray  # (n_ticks,) 每 tick 的收斂 steps
│
├── reward_mean:        float   # 最終 tick 的 mean reward
├── reward_trend:       np.ndarray  # (n_ticks,) 每 tick 的 final reward
├── reward_improvement: float   # 第一 tick vs 最後 tick reward delta
│
├── qpos_discontinuity_mean: float  # tick 邊界 qpos jump 平均
├── qpos_discontinuity_max:  float
├── ctrl_discontinuity_mean: float
│
├── reward_variance:    float   # across-tick reward variance（穩定性）
└── cost_breakdown:     dict    # {component_name: mean_cost}
```

**介面定義**:
```
compute_metrics(data: TrajectoryData) → ExperimentMetrics
compute_tick_boundary_discontinuity(data: TrajectoryData) → np.ndarray
compute_reward_variance(data: TrajectoryData) → float
```

**依賴**: `numpy`, `core/loader.py`, `spider.postprocess.get_success_rate` (compute_object_tracking_error)

---

### 2.5 `core/config_builder.py` — Config Assembler

**職責**: 把 UI 的 widget 值組裝成 `spider.config.Config` dataclass

**輸入**: `dict` of widget values  
**輸出**: `spider.config.Config`

**介面定義**:
```
build_config(ui_values: dict) → Config
config_to_dict(config: Config) → dict
config_to_json(config: Config) → str
load_config_from_json(json_str: str) → Config
get_default_config() → Config
diff_configs(config_a: Config, config_b: Config) → dict  # 只回傳不同的欄位
```

**依賴**: `spider.config` (Config, process_config, load_config_yaml, filter_config_fields)

---

### 2.6 `ui/config_panel.py` — Sidebar Config Panel

**職責**: 渲染四級旋鈕，回傳 widget values dict

**介面定義**:
```
render_config_panel(default_config: Config) → dict
render_task_selector() → dict  # dataset_name, robot_type, embodiment_type, task, data_id
```

**UI 佈局詳見 Section 5**

---

### 2.7 `ui/charts.py` — Chart Library

**職責**: 包裝所有 plotly 圖表為可複用函數，輸入 ndarray，輸出 `plotly.Figure`

**介面定義（對應 viz 腳本）**:
```
# v01
plot_reward_convergence(data: TrajectoryData) → Figure
plot_improvement_heatmap(data: TrajectoryData) → Figure
plot_reward_across_ticks(data: TrajectoryData) → Figure

# v02
plot_cost_decomposition(data: TrajectoryData) → Figure

# v03
plot_joint_timeseries(data: TrajectoryData, joint_idx: int) → Figure
plot_tick_discontinuity(data: TrajectoryData) → Figure

# v04
plot_ctrl_timeseries(data: TrajectoryData) → Figure
plot_ctrl_effort_vs_optsteps(data: TrajectoryData) → Figure

# v05
plot_3d_trajectory(data: TrajectoryData) → Figure
plot_bimanual_paths(data: TrajectoryData) → Figure

# v06
plot_reward_across_rollouts(data_list: List[TrajectoryData], labels: List[str]) → Figure
plot_cross_task_comparison(records: List[ExperimentRecord]) → Figure

# v07
plot_multi_experiment_reward(records: List[ExperimentRecord]) → Figure
plot_multi_experiment_optsteps(records: List[ExperimentRecord]) → Figure
plot_multi_experiment_cost_breakdown(records: List[ExperimentRecord]) → Figure
```

**依賴**: `plotly.express`, `plotly.graph_objects`, `core/loader.py`

---

### 2.8 `viz_adapters/` — Viz Script Adapters

**職責**: 把現有 `spider/notes/spider/viz/0X_*.py` 的繪圖邏輯包成函數，供 `ui/charts.py` 呼叫

**設計原則**:
- 每個 adapter 對應一個 viz 腳本
- 只做「matplotlib Figure → plotly Figure」的轉換，不重寫邏輯
- 若原腳本已用 plotly，直接 re-export

**依賴**: 各 viz 腳本, `matplotlib`, `plotly`

---

## 3. Data Flow 圖

```
╔══════════════════════════════════════════════════════════════════════╗
║                        USER INTERACTION                              ║
╚══════════════════════════════════════════════════════════════════════╝
         │                              │
         ▼ (調參數、選 task)              ▼ (選已有實驗)
╔══════════════════╗           ╔══════════════════════╗
║  ui/config_panel ║           ║ ui/experiment_selector║
╚══════════════════╝           ╚══════════════════════╝
         │                              │
         ▼                              ▼
╔══════════════════════╗       ╔══════════════════════╗
║ core/config_builder  ║       ║   core/registry.py   ║
║ Config dataclass     ║       ║   (SQLite query)      ║
╚══════════════════════╝       ╚══════════════════════╝
         │                              │
         ▼                              ▼
╔══════════════════════╗       ╔══════════════════════╗
║   core/runner.py     ║       ║   core/loader.py     ║
║ subprocess launch    ║       ║ load npz → cache     ║
║ examples/run_mjwp.py ║       ╚══════════════════════╝
╚══════════════════════╝                │
         │                              ▼
         │ (write)           ╔══════════════════════╗
         ▼                   ║   core/analyzer.py   ║
╔══════════════════════╗     ║ compute 4 metrics    ║
║  trajectory_mjwp.npz ║     ╚══════════════════════╝
╚══════════════════════╝                │
         │                              │
         └──────────────────────────────┘
                          │
                          ▼
               ╔══════════════════════╗
               ║   viz_adapters/      ║
               ║  (wrap viz scripts)  ║
               ╚══════════════════════╝
                          │
                          ▼
               ╔══════════════════════╗
               ║    ui/charts.py      ║
               ║  plotly Figures      ║
               ╚══════════════════════╝
                          │
                          ▼
               ╔══════════════════════╗
               ║   Streamlit Pages    ║
               ║  st.plotly_chart()   ║
               ╚══════════════════════╝
                          │
                          ▼
               ╔══════════════════════╗
               ║  Browser (WSL2 port  ║
               ║  forwarding :8501)   ║
               ╚══════════════════════╝
```

### 狀態同步流程（Runner ↔ UI）

```
User clicks "Run" 
    → registry.create_experiment()
    → runner.launch_experiment()  [Popen, detached]
    → st.session_state["active_run_id"] = experiment_id
    → st.rerun() loop every 5s
        → registry.get_experiment(id).status
        → runner.get_log_tail(id)
        → runner.estimate_progress(id)
    → status = "done" → loader.load_trajectory()
    → analyzer.compute_metrics()
    → render charts
```

---

## 4. 頁面規劃

### 4.1 `pages/1_Overview.py` — Dashboard Overview

**目的**: 鳥瞰所有實驗，快速找到感興趣的 run

**Layout**:
```
┌────────────────────────────────────────────────────────┐
│ [篩選欄] status: All▾  task: All▾  robot: All▾  [搜尋] │
├────────────────────────────────────────────────────────┤
│ 📊 Summary Cards                                        │
│  Total Experiments: 42  |  Running: 1  |  Done: 40     │
├──────────────────────────┬─────────────────────────────┤
│ 實驗列表（sortable table）│  快速預覽（右側 panel）       │
│ ID | Task | Robot | Status│  選中實驗的四大指標 sparkline │
│ ── | ─── | ───── | ─────  │  opt_steps trend            │
│ e1 | tea  | xhand | done  │  reward trend                │
│ e2 | pick | allgro| running│  [→ 詳細分析] [→ 比較]      │
│ ...                       │                             │
└──────────────────────────┴─────────────────────────────┘
```

**使用模組**: `core/registry.py`, `core/loader.py`, `core/analyzer.py`, `ui/metric_cards.py`

---

### 4.2 `pages/2_Single_Run.py` — Single Run Analysis

**目的**: 深入分析單一實驗的所有面向

**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│ Sidebar: [實驗選擇器] or [Tick 滑桿 0~n_ticks]           │
├────────────────────────────┬────────────────────────────┤
│ 四大指標卡片（頂部一排）                                   │
│  opt_steps↓ | reward↑ | discontinuity↓ | variance↓     │
├────────────────────────────┬────────────────────────────┤
│  Tab: Reward  │ Cost │ Trajectory │ Control │ 3D         │
│                                                          │
│  [Plotly Chart - 依 tab 切換]                            │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ Raw Data Inspector (expander, 預設關閉)                   │
│  npz shape info, tick selector, raw array preview        │
└─────────────────────────────────────────────────────────┘
```

**Tab 對應 viz**:
- Reward → `v01_reward.py` (convergence, heatmap, across-ticks)
- Cost → `v02_cost.py` (decomposition)
- Trajectory → `v03_trajectory.py` (joint timeseries, discontinuity)
- Control → `v04_control.py` (ctrl timeseries, effort)
- 3D → `v05_3d.py` (receding horizon, bimanual)

**使用模組**: `core/loader.py`, `core/analyzer.py`, `ui/charts.py`, `ui/metric_cards.py`

---

### 4.3 `pages/3_Compare.py` — Compare Experiments

**目的**: 多組實驗疊加比較，找出最佳參數組合

**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│ [多選實驗] Experiment A: ▾  B: ▾  C: ▾  [+ Add]         │
│ [自動對齊模式] by task | by robot | manual               │
├─────────────────────────────────────────────────────────┤
│ Config Diff Table（只顯示不同的欄位）                     │
│  Param          | Exp A  | Exp B  | Exp C              │
│  temperature    | 0.3    | 0.5    | 0.3                │
│  num_samples    | 2048   | 2048   | 4096               │
├─────────────────────────────────────────────────────────┤
│ Metrics Comparison                                       │
│  [Bar chart: opt_steps / reward / discontinuity]         │
├─────────────────────────────────────────────────────────┤
│ Overlay Charts                                           │
│  Tab: Reward Trend │ opt_steps │ Cost Breakdown          │
│  [Plotly: 多線疊加，每條線一個實驗]                        │
└─────────────────────────────────────────────────────────┘
```

**使用模組**: `core/registry.py`, `core/loader.py`, `core/analyzer.py`, `core/config_builder.py` (diff_configs), `ui/charts.py`

---

### 4.4 `pages/4_New_Run.py` — Run New Experiment

**目的**: 調參、命名、啟動實驗、監控進度

**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│ Sidebar: Config Panel（四級旋鈕，詳見 Section 5）          │
├─────────────────────────────────────────────────────────┤
│ 實驗命名  [text input] + auto-suggest from params        │
│ Tags      [text input, comma separated]                  │
├──────────────────────────┬──────────────────────────────┤
│ Config Summary（JSON view）│  預估資源                    │
│  只顯示非預設值            │  GPU: RTX 3060 Ti           │
│                           │  預估時間: ~25 min           │
│                           │  Output: /path/to/output/   │
├─────────────────────────────────────────────────────────┤
│                [▶ Launch Experiment]                      │
├─────────────────────────────────────────────────────────┤
│ 執行中（條件顯示）                                        │
│  Progress: ████████░░░░ 67%  Tick 48/72                 │
│  Elapsed: 16:42  |  ETA: ~8 min                         │
│  [Log Viewer - 滾動, 最新 50 行]                          │
│  [■ Kill Run]                                            │
├─────────────────────────────────────────────────────────┤
│ 完成後                                                   │
│  ✅ Done in 24:33  →  [View Results] [Compare with...]  │
└─────────────────────────────────────────────────────────┘
```

**使用模組**: `ui/config_panel.py`, `core/config_builder.py`, `core/runner.py`, `core/registry.py`

---

## 5. Config Panel 設計

### Sidebar 佈局

```
┌─────────────────────────┐
│ 🎯 Task Selection        │
│  Dataset:  [gigahand ▾] │
│  Robot:    [xhand    ▾] │
│  Embodiment:[bimanual▾] │
│  Task:     [p36-tea  ▾] │
│  Data ID:  [0        ▾] │
├─────────────────────────┤
│ ⚡ Level 1 — Safe        │
│  num_samples     [2048] │
│  max_num_iters   [  16] │
│  temperature     [0.30] │
│  improve_thresh  [0.01] │
│  seed            [   0] │
├─────────────────────────┤
│ 🔧 Level 2 — Noise  ▼   │  ← 預設折疊
│  joint_noise_scale      │
│  pos_noise_scale        │
│  rot_noise_scale        │
│  final_noise_scale      │
│  first_ctrl_noise_scale │
│  last_ctrl_noise_scale  │
├─────────────────────────┤
│ ⏱ Level 3 — Time    ▼   │  ← 預設折疊
│  sim_dt, ctrl_dt        │
│  horizon, knot_dt       │
├─────────────────────────┤
│ 🤝 Level 4 — Contact ▼  │  ← 預設折疊
│  contact_guidance [☐]   │
│  guidance_decay_ratio   │
│  gibbs_sampling   [☐]   │
├─────────────────────────┤
│ 💰 Reward Weights    ▼   │  ← 預設折疊
│  base_pos_rew_scale     │
│  base_rot_rew_scale     │
│  joint_rew_scale        │
│  vel_rew_scale          │
│  contact_rew_scale      │
├─────────────────────────┤
│ [↺ Reset to Default]    │
│ [💾 Save as Preset]     │
│ [📂 Load Preset]        │
└─────────────────────────┘
```

### Widget 類型對應

| 參數類型 | Widget | 備註 |
|----------|--------|------|
| `int` (大範圍) | `st.slider` + `st.number_input` | 並排，互動同步 |
| `float` (0~1) | `st.slider(step=0.01)` | |
| `float` (小值如 vel_rew_scale) | `st.number_input(format="%.6f")` | slider 無意義 |
| `bool` | `st.checkbox` | |
| `str` (enum) | `st.selectbox` | |
| `int` (small) | `st.number_input` | |

### Preset 格式（JSON）

```json
{
  "preset_name": "baseline_xhand",
  "created_at": "2026-04-06T00:00:00Z",
  "description": "Default xhand baseline on gigahand",
  "config": {
    "num_samples": 2048,
    "temperature": 0.3,
    ...
  }
}
```

儲存於 `dashboard/presets/` 目錄，命名為 `{preset_name}.json`

---

## 6. Experiment Registry

### 選型：SQLite

**理由**:
- 本地單機，無需網路
- 支援 SQL 查詢（按 task/status/date 篩選）
- Python stdlib 內建，零依賴
- 檔案單一，易備份
- 支援 concurrent read（Streamlit rerun 不衝突）

### Registry 操作層設計

```
ExperimentRecord:
├── id: str             # UUID4
├── name: str
├── status: Literal["pending", "running", "done", "failed"]
├── created_at: datetime
├── config: Config      # 從 config_json 反序列化
├── task: str
├── robot_type: str
├── dataset_name: str
├── data_id: int
├── output_dir: Path | None
├── pid: int | None
├── duration_s: float | None
├── error_msg: str | None
└── tags: List[str]

RegistryDB:
├── __init__(db_path: Path)  # 自動建表
├── create(config, name, tags) → ExperimentRecord
├── update_status(id, status, **kwargs)
├── list(filters: dict, limit: int) → List[ExperimentRecord]
├── get(id) → ExperimentRecord
├── delete(id)
└── search(query: str) → List[ExperimentRecord]  # 全文搜尋 name/tags
```

### Output 目錄命名規則

```
{SPIDER_ROOT}/outputs/{task}/{robot_type}/{experiment_id}/
    ├── trajectory_mjwp.npz
    ├── run.log
    └── config.json  # 該次實驗的完整 Config snapshot
```

### 實驗 ID 命名策略

- UUID 做主鍵（確保唯一）
- 人類可讀名稱（human name）自動生成：`{task}_{robot}_{timestamp}_{short_uuid}`
- 例：`p36-tea_xhand_20260406-0156_a3f2`

---

## 7. 實驗執行策略

### 核心挑戰

- Streamlit 每次 user interaction 都會 rerun 整個腳本
- 長達 25 min 的 process 不能跑在 Streamlit thread 裡
- WSL2 無 display server，MuJoCo 需用 headless rendering

### 解決方案：Detached Subprocess

```
┌─────────────────┐     Popen(detached)    ┌─────────────────────┐
│  Streamlit App  │ ─────────────────────→ │  runner_subprocess.py│
│  (main thread)  │                         │  (獨立 Python process)│
│                 │ ←─────────────────────  │  - import spider    │
│  poll registry  │   registry.db (SQLite)  │  - main(config)     │
│  every 5s       │   run.log (file)        │  - write log        │
└─────────────────┘                         └─────────────────────┘
```

### `runner_subprocess.py` 腳本設計

這是 runner 啟動的 subprocess 的入口點（不是 Streamlit 的一部分）：

```
Input: experiment_id (argv[1])
Process:
  1. load config from registry
  2. update status = "running"
  3. setup MuJoCo headless (no display)
  4. call examples/run_mjwp.py:main(config)
  5. on success: update status = "done", set output_dir, set duration
  6. on exception: update status = "failed", set error_msg
Output: trajectory_mjwp.npz in output_dir
```

### WSL2 Headless MuJoCo

```bash
# .streamlit/config.toml 或 runner 啟動前設定
MUJOCO_GL=egl  # 或 osmesa（軟體渲染）
```

- `MUJOCO_GL=egl`：需要 GPU，效能好，WSL2 + NVIDIA driver 通常可用
- `MUJOCO_GL=osmesa`：CPU 軟體渲染，fallback 方案
- Runner 啟動時自動偵測，egl 失敗則 fallback osmesa

### Progress Tracking

**方法一（優先）**: Log parsing
- `run_mjwp.py` 每 tick 印 `[Tick {i}/{total}]`
- runner 解析 `run.log` 最後一行，計算進度百分比

**方法二（fallback）**: npz size heuristic
- `trajectory_mjwp.npz` 大小隨 tick 線性增長
- 根據已完成 run 的 npz 大小 vs tick 數估算

**方法三（精確）**: Checkpoint npz
- 若 SPIDER 支援，每 N ticks 寫一次 partial npz
- Dashboard 可提前顯示已完成 tick 的 metrics

### Streamlit Polling 機制

```
if "active_run_id" in st.session_state:
    experiment = registry.get(st.session_state["active_run_id"])
    
    if experiment.status == "running":
        progress = runner.estimate_progress(experiment.id)
        st.progress(progress)
        st.text(runner.get_log_tail(experiment.id))
        time.sleep(5)
        st.rerun()  # 5 秒後重新渲染
    
    elif experiment.status == "done":
        # 載入結果，顯示圖表
        data = loader.load_trajectory(experiment.output_dir)
```

### 並發限制

- 同時只允許 1 個 running experiment（GPU 記憶體限制）
- Launch 前檢查 registry，若有 running 實驗則 warn user
- 提供 "Kill current run" 選項（`os.kill(pid, SIGTERM)`）

---

## 8. 可複用模組對照表

| Dashboard 功能 | SPIDER 模組 | 函數/類別 |
|---------------|------------|----------|
| 組裝 Config | `spider.config` | `Config`, `process_config()`, `load_config_yaml()`, `filter_config_fields()` |
| 載入 npz | `spider.io` | `load_data()`, `get_processed_data_dir()` |
| 軌跡插值 | `spider.interp` | `interp()`, `get_slice()` |
| 跑實驗（rollout） | `spider.optimizers.sampling` | `make_rollout_fn()`, `make_optimize_fn()` |
| 跑實驗（主流程） | `examples/run_mjwp.py` | `main(config)` |
| MuJoCo 環境 | `spider.simulators.mjwp` | `setup_env()`, `step_env()`, `sync_env()`, `get_reward()` |
| 成功率計算 | `spider.postprocess.get_success_rate` | `compute_object_tracking_error()` |
| Reward 收斂圖 | `viz/01_*.py` | plot functions（wrap 為 adapter） |
| Cost 分解圖 | `viz/02_*.py` | plot functions |
| 軌跡 vs ref | `viz/03_*.py` | plot functions |
| 控制信號圖 | `viz/04_*.py` | plot functions |
| 3D 軌跡 | `viz/05_*.py` | plot functions |
| 跨 rollout 比較 | `viz/06_*.py` | plot functions |
| 多實驗比較 | `viz/07_*.py` | plot functions |

### npz 欄位 → Dashboard 功能對照

| npz 欄位 | 型狀 | 用於指標 | 用於圖表 |
|----------|------|----------|----------|
| `opt_steps` | (n_ticks, 1) | ✅ opt_steps mean/std | v04, v07 |
| `rew_mean` | (n_ticks, max_iter) | ✅ reward level | v01, v06, v07 |
| `rew_min/max` | (n_ticks, max_iter) | reward bound | v01 |
| `qpos` | (n_ticks, ctrl_steps, nq) | ✅ discontinuity | v03, v05 |
| `ctrl` | (n_ticks, ctrl_steps, nu) | discontinuity | v04 |
| `trace_cost` | (n_ticks, max_iter, n_comp) | - | v02, v07 |
| `qpos_dist_mean` | (n_ticks, max_iter) | ✅ tracking error | v07 |
| `trace_ref` | (n_ticks, 1, 1, h, n_sites, 3) | - | v03, v05 |

---

## 9. 啟動與部署

### Streamlit 設定（`.streamlit/config.toml`）

```toml
[server]
port = 8501
headless = true          # WSL2 必須
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "dark"            # 實驗室深色主題
```

### 啟動腳本（`run_dashboard.sh`）

```bash
#!/bin/bash
cd /path/to/spider
export MUJOCO_GL=egl    # 優先用 EGL
export PYTHONPATH="/path/to/spider:$PYTHONPATH"
streamlit run dashboard/app.py
```

### 從 Windows/Mac 存取

```
WSL2: streamlit run app.py --server.port 8501
Windows: http://localhost:8501  (WSL2 port forwarding 自動)
Mac (SSH tunnel): ssh -L 8501:localhost:8501 user@wsl-host
```

---

## 10. 實作優先順序建議

### Phase 1 — 讀取 + 視覺化（最小可用版）
1. `core/loader.py` — npz 讀取
2. `core/analyzer.py` — 四大指標計算
3. `ui/metric_cards.py` — 指標卡片
4. `viz_adapters/v01_reward.py` — 第一個圖表 wrap
5. `pages/2_Single_Run.py` — 單頁面跑通

### Phase 2 — Registry + 管理
6. `core/registry.py` — SQLite schema + CRUD
7. `ui/experiment_selector.py` — 選擇器 widget
8. `pages/1_Overview.py` — 實驗列表

### Phase 3 — Config Panel + Runner
9. `ui/config_panel.py` — 四級旋鈕
10. `core/config_builder.py` — Config 組裝
11. `core/runner.py` — Background subprocess
12. `pages/4_New_Run.py` — 完整啟動流程

### Phase 4 — 比較 + 完整 viz
13. 剩餘 `viz_adapters/v02~v07_*.py`
14. `ui/charts.py` — 完整圖表庫
15. `pages/3_Compare.py` — 多實驗比較

---

*文件版本 v1.0 — 設計階段，後續實作時各 sub-agent 應先閱讀本文件，再認領對應模組。*
