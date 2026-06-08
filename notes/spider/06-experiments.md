# 06 — 實驗與參數探索

## 1. 實驗環境

- **GPU**: NVIDIA GeForce RTX 3060 Ti (8 GiB, sm_86)
- **平台**: WSL2, CUDA 12.8
- **Warp**: ⚠️ 初期實驗使用 1.11.0.dev（有 actuator gain bug），後期修正為 **1.12.1**（stable）
- **Headless**: `show_viewer=false save_video=false`（WSL2 無 display server）
- **任務**: gigahand / xhand / bimanual / p36-tea（17 ticks, 676 sim_steps）
- **Baseline**: repo 附帶的 `trajectory_mjwp.npz`（已備份為 `_baseline.npz`）

---

## 2. Baseline Config 解析

從 `config.yaml` 讀出 baseline 的關鍵參數：

| 參數 | Baseline 值 | Code 預設值 | 差異 |
|---|---|---|---|
| `temperature` | **0.1** | 0.3 | 更銳利的 softmax，top sample 主導 |
| `max_num_iterations` | **32** | 16 | 允許更多 iteration |
| `improvement_threshold` | **0.001** | 0.01 | 更嚴格的 early stopping |
| `num_samples` | 2048 | 2048 | 相同 |
| `nconmax_per_env` | 120 | 100 | baseline 稍高 |

其餘參數（noise scale、reward weights、timing）均與預設一致。

---

## 3. 實驗記錄

### 實驗 1: 預設 config（T=0.3, th=0.01, iter=16）

```bash
uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
  num_samples=2048 seed=0 show_viewer=false save_video=false
```

| 指標 | 值 |
|---|---|
| opt_steps | mean=1.8, max=8, total=31 |
| reward mean | −0.499 |
| reward min | −1.202 |
| tick disc mean | 0.00360 |
| 耗時 | ~3 min |

**觀察**: 大部分 tick 只跑 1 步就 early stop。reward 比 baseline 差很多。

### 實驗 2: 收緊 early stopping（T=0.3, th=0.001, iter=16）

**假設**: reward 差是因為 early stopping 太激進，多跑幾輪應該改善。

```bash
uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
  num_samples=2048 seed=0 improvement_threshold=0.001 \
  show_viewer=false save_video=false
```

| 指標 | 值 |
|---|---|
| opt_steps | mean=5.1, max=32, total=87 |
| reward mean | −0.523 |
| reward min | −1.215 |
| tick disc mean | 0.00354 |
| 耗時 | ~5 min |

**結果**: opt_steps 上升了（1.8→5.1），但 reward 反而更差（−0.499→−0.523）。**假設被否定。** 多跑 iteration 在 T=0.3 下無效——退火後 noise 太小，optimizer 卡在 local minimum 裡磨。

### 實驗 3: 復現 baseline config（T=0.1, th=0.001, iter=32）

```bash
uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
  num_samples=2048 seed=0 temperature=0.1 max_num_iterations=32 \
  improvement_threshold=0.001 show_viewer=false save_video=false
```

| 指標 | 值 |
|---|---|
| opt_steps | mean=4.0, max=10, total=68 |
| reward mean | −0.405 |
| reward min | −0.767 |
| tick disc mean | 0.00374 |
| 耗時 | ~5 min |

**結果**: reward 從 −0.52 改善到 −0.40，最差 tick 從 −1.21 改善到 −0.77。**Temperature 是主要因子。**

---

## 4. 綜合比較

| 指標 | Baseline | Exp1 (預設) | Exp2 (th↓) | Exp3 (T↓) |
|---|---|---|---|---|
| temperature | 0.1 | 0.3 | 0.3 | **0.1** |
| threshold | 0.001 | 0.01 | 0.001 | 0.001 |
| max_iter | 32 | 16 | 16 | 32 |
| opt_steps mean | 9.8 | 1.8 | 5.1 | 4.0 |
| **reward mean** | **−0.264** | −0.499 | −0.523 | **−0.405** |
| reward min | −0.902 | −1.202 | −1.215 | −0.767 |
| tick disc mean | 0.00371 | 0.00360 | 0.00354 | 0.00374 |

### 關鍵結論

1. **Temperature 是 MPPI 最重要的超參數**（至少在 contact-rich 任務中）
   - T=0.3（民主投票）vs T=0.1（精英獨裁）
   - 在 non-convex feasible set 上，多數好 sample 的 convex combination 可能落在不可行區域
   - T 越低，結果越接近 best sample，繞過了 convex combination 封閉性問題

2. **Early stopping threshold 不是瓶頸**
   - 收緊 threshold 增加 opt_steps 但不改善 reward
   - 問題不在「跑不夠久」，而在「加權方式錯了」

3. **Baseline 未完全復現（−0.264 vs −0.405）**
   - 可能原因：GPU 浮點非確定性、Warp/MuJoCo 版本差異、baseline 可能是多次跑取最佳

---

## 5. 系統化調參框架

### 控制變因法（One-Factor-at-a-Time）

固定 baseline config 作為起點，每次只改一個參數：

```
Base: T=0.1, th=0.001, iter=32, samples=2048, seed=0
```

#### Sweep 1: Temperature（策略銳度）
```bash
for T in 0.05 0.1 0.2 0.3 0.5; do
  uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
    temperature=$T num_samples=2048 seed=0 \
    max_num_iterations=32 improvement_threshold=0.001 \
    show_viewer=false save_video=false
  cp .../trajectory_mjwp.npz .../trajectory_mjwp_T${T}.npz
done
```

#### Sweep 2: Sample 數量（搜索密度）
```bash
for N in 256 512 1024 2048 4096; do
  uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
    num_samples=$N temperature=0.1 seed=0 \
    max_num_iterations=32 improvement_threshold=0.001 \
    show_viewer=false save_video=false
  cp .../trajectory_mjwp.npz .../trajectory_mjwp_N${N}.npz
done
```

#### Sweep 3: Noise scale（搜索幅度）
```bash
for JS in 0.05 0.1 0.15 0.2 0.3; do
  uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
    joint_noise_scale=$JS temperature=0.1 seed=0 \
    max_num_iterations=32 improvement_threshold=0.001 \
    show_viewer=false save_video=false
  cp .../trajectory_mjwp.npz .../trajectory_mjwp_JS${JS}.npz
done
```

#### Sweep 4: 跨 seed 穩定性（每組最佳 config）
```bash
for S in 0 1 2 3 4; do
  uv run examples/run_mjwp.py dataset_name=gigahand task=p36-tea \
    seed=$S temperature=0.1 num_samples=2048 \
    max_num_iterations=32 improvement_threshold=0.001 \
    show_viewer=false save_video=false
  cp .../trajectory_mjwp.npz .../trajectory_mjwp_seed${S}.npz
done
```

### 分析腳本模板

每次 sweep 完，用統一腳本比較：

```python
import numpy as np, glob

def summarize(path):
    d = np.load(path)
    opt = d['opt_steps'].squeeze()
    rew = d['rew_mean']
    n = rew.shape[0]
    final = np.array([rew[t, int(opt[t])-1] if opt[t]>0 else rew[t,0] for t in range(n)])
    qpos = d['qpos']
    discs = np.array([np.abs(qpos[t+1,0,:]-qpos[t,-1,:]).mean() for t in range(n-1)])
    return {
        'opt_mean': opt.mean(), 'opt_max': int(opt.max()),
        'rew_mean': final.mean(), 'rew_min': final.min(),
        'disc_mean': discs.mean(), 'disc_max': discs.max(),
    }

for f in sorted(glob.glob('.../trajectory_mjwp_*.npz')):
    s = summarize(f)
    print(f'{f.split("_")[-1]:>12s}  rew={s["rew_mean"]:.4f}  opt={s["opt_mean"]:.1f}  disc={s["disc_mean"]:.5f}')
```

### 調參順序建議

1. **Temperature** → 確定最佳策略銳度（預期 0.05-0.15 最好）
2. **num_samples** → 找到品質-速度的 sweet spot（預期 1024 夠用）
3. **noise_scale** → 微調搜索幅度（可能任務相關）
4. **跨 seed** → 驗證最佳 config 的穩定性

每一步拿最佳結果作為下一步的 base，不回頭。

---

## Phase 6 完成條件

- [x] 有一個能在 <10 行改參數就跑新實驗的流程
- [ ] 完成至少一組 sweep 並記錄結果
- [ ] 找到比 baseline 更好（或等同）的 config

---

## 6. Warp 版本對實驗結果的影響

### 發現經過
- 所有 Section 3-4 的實驗（Exp1-3 + baseline 比較）都在 Warp 1.11.0-dev 下跑的
- 物件（茶壺）在所有實驗中都沒有被抬起（object z 恆定 ~0.08）
- 一度以為是 num_samples、contact guidance clamp、reward function 的問題
- 最終確認是 **Warp 1.11.0-dev 的 GPU kernel bug**：無法正確讀取 runtime 更新的 `actuator_gainprm`

### Warp 版本對路徑輸出的影響

| 面向 | Warp 1.11.0-dev | Warp 1.12.1 (stable) |
|---|---|---|
| Object z trajectory | 恆定 ~0.078，無抬起 | 正確追蹤 reference（peak 0.24） |
| Actuator gain decay | 失效（kp 值寫入但 GPU 不讀取） | 正常（kp 10→0 指數衰減） |
| Finger-object contact | 手指到位但無力閉合 | 正確形成 force closure |
| Reward profile | 看似正常（qpos tracking 改善） | 真正反映任務完成度 |
| 路徑品質 | 手指軌跡「看起來對」但物件不動 | 手指+物件軌跡都正確 |

### 影響範圍
- Section 3-4 的 reward 比較（Exp1-3 vs baseline）在 Warp 1.11-dev 下仍然有效——因為都是 qpos tracking reward，不涉及物件狀態
- 但 **任何涉及物件行為的結論都需要在 Warp 1.12.1 下重新驗證**
- Multi-task 30 runs（3 methods × 10 tasks）已在修正後重跑

### 教訓
- 模擬器版本是最底層的依賴，bug 可以讓所有上層分析都建立在錯誤基礎上
- SPIDER 的 reward 是純 qpos tracking，不包含 object state → reward 改善不代表任務成功
- 永遠先確認 reference trajectory 的 ground truth 行為，再分析實驗結果
