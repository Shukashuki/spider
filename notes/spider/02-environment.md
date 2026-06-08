# 02 — 環境與重現

## 目標

從乾淨的 WSL2 環境，一鍵跑通 SPIDER demo（`p36-tea` / gigahand / xhand / bimanual）。

---

## 硬體需求

| 項目 | 最低需求 | 我們的環境 |
|---|---|---|
| GPU | NVIDIA GPU, CUDA ≥12.0, ≥8GB VRAM | RTX 3060 Ti, 8GB, CUDA 12.8 |
| RAM | ≥16GB | 23GB |
| Disk | ≥10GB（repo + dataset） | 786GB available |
| OS | Linux / WSL2 | WSL2 (Ubuntu, kernel 6.6.114.1-microsoft-standard-WSL2) |
| Python | 3.12 | 3.12.3 |
| Warp | **1.12.1** (stable) | ⚠️ 1.11.0-dev 有 GPU sync bug，見下方 |

> ⚠️ **Warp 版本是最關鍵的依賴**。1.11.0.dev20251120 有 GPU kernel 無法正確讀取更新後的 actuator gain 值的 bug，導致 gain decay schedule 失效、物件無法被抬起。必須使用 requirements.txt pinned 的 mujoco-warp commit 對應的 Warp 1.12.1。

---

## 完整重現步驟

### 0. 前置工具

```bash
# git-lfs（WSL2 預設沒有）
sudo apt update && sudo apt install -y git-lfs python3.12-dev
git lfs install
```

> ⚠️ `python3.12-dev` 必須裝，否則 Triton JIT compile 會找不到 `Python.h`
> 錯誤訊息: `fatal error: Python.h: No such file or directory` at `/tmp/.../cuda_utils.c`

### 1. Clone repo + 安裝依賴

```bash
git clone https://github.com/facebookresearch/spider.git
cd spider

# 安裝 uv（如果還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安裝所有 Python 依賴（181 packages）
# ⚠️ 這會安裝 requirements.txt pinned 的 mujoco-warp (commit 5d5a645) → Warp 1.12.1
# 不要用舊的 Warp 1.11.0-dev，會導致 actuator gain schedule 失效
uv sync
```

### 2. 下載 example dataset

```bash
# ~4.06 GB, 5068 LFS files
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

> ⚠️ 需要 git-lfs，否則 LFS files 只會是 pointer（~130 bytes 的文字檔）
> 已知問題: 2 個 mp4 不是 LFS pointer（不影響 demo）:
> - `processed/arctic/dex3_hand/bimanual/box-30-230/0/trajectory_dexmachina.mp4`
> - `processed/arctic/xhand/bimanual/box-30-230/0/trajectory_dexmachina.mp4`

### 3. 跑 demo

```bash
export TASK=p36-tea
export HAND_TYPE=bimanual
export DATA_ID=0
export ROBOT_TYPE=xhand
export DATASET_NAME=gigahand

uv run python examples/run_mjwp.py \
  +override=${DATASET_NAME} \
  task=${TASK} \
  data_id=${DATA_ID} \
  robot_type=${ROBOT_TYPE} \
  embodiment_type=${HAND_TYPE} \
  show_viewer=false \
  save_video=false
```

---

## WSL2 特有問題 & 解法

### 問題 0: ⚠️ Warp 版本 — 最重要的坑

**症狀**: 所有實驗看起來都能跑，reward 也在改善，但物件永遠不會被抬起（object z 恆定）

**根因**: Warp 1.11.0.dev20251120 的 GPU kernel 無法正確讀取 runtime 更新的 `actuator_gainprm`。SPIDER 的核心機制是 actuator gain decay schedule（kp 從 10.0 指數衰減到 0），讓物件從 PD 控制逐步釋放給手指物理接觸。這個 bug 讓 gain schedule 完全失效。

**證據**:
| Warp 版本 | Teapot Z max | Z range | 成功抬起? |
|---|---|---|---|
| Paper SPIDER | 0.2434 | 0.1649 | ✅ ~16cm |
| 1.11.0-dev | 0.0868 | 0.0083 | ❌ |
| **1.12.1** (stable) | **0.2387** | **0.1611** | **✅ ~16cm** |

**解法**: 使用 `uv sync` 安裝 requirements.txt pinned 的 mujoco-warp commit → 自動得到 Warp 1.12.1。如果已有舊環境，重建 conda env：
```bash
conda create -n spider python=3.12
conda activate spider
pip install -r requirements.txt  # 或 uv sync
```

**被這個 bug 誤導的假設**（全部排除）:
- ❌ num_samples 太少（1024 vs 2048）
- ❌ contact guidance clamp ±0.01 太小
- ❌ reward function 缺少 object state
- ❌ GPU sync 代碼有問題

### 問題 1: `gladLoadGL error` / GLFW crash

**症狀**: MuJoCo viewer 一啟動就 crash，因為 WSL2 沒有 display server（無 `DISPLAY`、`WAYLAND_DISPLAY`、`WAYLAND_SOCKET`）

**解法**: 加 Hydra overrides 禁用 viewer 和 video rendering
```
show_viewer=false save_video=false
```

Config 會自動 fallback 到 `dummy viewer`。

### 問題 2: Triton 找不到 `Python.h`

**症狀**: 
```
fatal error: Python.h: No such file or directory
/tmp/tmplrt340x4/cuda_utils.c:6:10
```

**原因**: Triton JIT compile CUDA utils 時需要 Python C header，WSL2 預設只裝 runtime 不裝 dev headers

**解法**:
```bash
sudo apt install python3.12-dev
```

### 問題 3: `nvidia-smi` 路徑

**症狀**: 標準 `nvidia-smi` 找不到

**解法**: WSL2 的 nvidia-smi 在 `/usr/lib/wsl/lib/nvidia-smi`，`libcuda.so` 在 `/usr/lib/wsl/lib/`。PyTorch / Warp 會自動找到，不需要手動設定。

### 問題 4: GPU memory 顯示 `[N/A]`

**症狀**: `nvidia-smi` 顯示 process 的 GPU memory 是 `[N/A]`

**原因**: WSL2 透過 `/dev/dxg` 做 GPU passthrough，memory accounting 不完整。**不影響功能**。

---

## 預期結果

```
Warp 1.12.1 initialized:
   CUDA Toolkit 12.8, Driver 12.8
   Devices:
     "cuda:0"   : "NVIDIA GeForce RTX 3060 Ti" (8 GiB, sm_86, mempool enabled)

...（Warp kernel compile，首次需要幾分鐘）...

Realtime rate: 0.01, plan time: 35.4s, sim_steps: 40/676, opt_steps: 7
...
Realtime rate: 0.11, plan time: 3.5s, sim_steps: 680/676, opt_steps: 1
Total time: 634.3s

Saved info to .../trajectory_mjwp.npz
Final object tracking error: pos=0.0643, quat=0.6682
```

- 首次跑: ~10 分鐘（含 Triton/Warp JIT compile）
- 之後跑（有 cache）: 預期更快
- Warp kernel cache: `~/.cache/warp/1.12.1/`

---

## 檔案結構（重現相關）

```
spider/
├── examples/
│   ├── run_mjwp.py              # 主入口
│   └── config/
│       └── default.yaml         # 預設 Hydra config (device: cuda:0, num_samples: 1024)
├── spider/                      # 核心模組
│   ├── config.py
│   ├── optimizers/sampling.py   # MPPI optimizer
│   ├── simulators/mjwp.py       # MuJoCo Warp backend
│   └── ...
├── example_datasets/            # git clone from HuggingFace
│   └── processed/gigahand/xhand/bimanual/p36-tea/0/
│       ├── trajectory_mjwp.npz  # ← 輸出在這裡
│       └── config.yaml
└── .venv/                       # uv 管理的虛擬環境
```

---

## Phase 2 完成條件

- [x] 從乾淨環境可以一鍵重現 Phase 1 的結果（上面的步驟 0-3）
- [x] 所有遇到的問題都記錄在案，包含解法（4 個 WSL2 特有問題）
