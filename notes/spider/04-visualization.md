[2026/4/5 下午 04:07] 怪異專家: 04 — 視覺化與理解驗證

概述

Phase 4 設計了 6 個系列、共 16 張視覺化圖，從 reward 收斂、tracking 分解、軌跡對比、控制信號、3D 空間軌跡、到跨 rollout/跨任務比較，逐層驗證 Phase 3 對 MPPI 的理論理解。主要資料來源為 trajectory_mjwp.npz（optimizer 輸出）與 trajectory_kinematic.npz（IK reference），任務為 p36-tea（bimanual, xhand）。

───

1. Reward 收斂分析（01 系列）

設計動機

驗證 MPPI 的 iterative optimization 是否真的在收斂，以及不同 tick 的收斂行為是否一致。

觀察

• 01a — Reward Convergence per Tick：選定 tick 0/5/10/16，reward max/mean/min 隨 iteration 上升。大部分 tick 在 1-3 個 iteration 內就收斂（early stopping 生效），只有特定 tick 需要更多步。
• 01b — Improvement Heatmap：mask 掉 opt_steps 之後的 iteration，可見 improvement 集中在前幾輪，後續幾乎為零——退火機制 (beta_traj^i) 配合 early stopping 運作正常。
• 01c — Cross-Tick Reward Trend：final iteration 的 reward 跨 tick 大致穩定，沒有明顯崩潰——在成功 rollout 中，receding horizon 維持了追蹤品質。

對理論的驗證

符合 Phase 3 對 MPPI 收斂機制的理解：退火降 noise → weighted average 精化 → early stopping 截斷。但也確認了「收斂 ≠ 最優」——只是 improvement 低於閾值就停了。

───

2. Tracking Reward 分解（02 系列）

設計動機

SPIDER 的 cost function 有 qpos（位置）和 qvel（速度）兩項。想知道哪個主導了 optimizer 行為。

觀察

• 02a — qpos/qvel Reward over Ticks：qpos reward 絕對值遠大於 qvel，兩者量級差 2-3 個 order of magnitude。
• 02b — Reward Component Ratio：qpos 佔 cost >99.7%，qvel 幾乎可忽略。這表示 optimizer 幾乎完全在追位置，速度只是象徵性地存在。
• 02c — Optimizer Effort：opt_steps 與 improvement 的對應關係——opt_steps 高的 tick 很少（p36-tea 只有個別 tick 跑滿），大部分 1-2 步就 early stop。

對理論的修正

原本以為位置和速度 tracking 同等重要。視覺化後發現 qvel 的權重設定讓它幾乎不影響優化方向。這是一個設計選擇：SPIDER 優先保證「手到對的位置」，而非「手以對的速度到達」。

───

3. 軌跡 vs Reference（03 系列）

設計動機

直接比較 MPPI 輸出的 sim trajectory 和 kinematic IK reference，驗證物理軌跡對 reference 的追蹤品質。

觀察

• 03a — Joint Timeseries：8 個代表性 joint 的 qpos vs reference。大部分 DOF 追蹤良好，偏差集中在特定 finger joints。error fill 顯示的偏差區域和 contact transition 位置相關。
• 03b — Tick Boundary Discontinuity：tick 邊界 |Δqpos| heatmap，mean 0.004, max 0.068。大部分 DOF 的 tick boundary 非常平滑——receding horizon 的 warm start 機制有效。Max 0.068 出現在 finger joints，可能是 contact transition 造成的 replanning 跳變。
• 03c — Per-Tick qpos Range：每個 tick 內 qpos 的變化幅度。wrist DOF 變化大（需要移動手到新位置），finger DOF 變化小（精細調整）。

對理論的驗證

Receding horizon 的 shift + warm start 確實維持了 tick 間連續性。不連續性集中在 contact transition 附近，符合 Phase 3 「contact boundary 造成映射不連續」的理論預測。

───

4. 控制信號分析（04 系列）

設計動機

軌跡好 ≠ 控制信號好。想知道 actuator ctrl 是否平滑，以及控制努力和優化難度的關係。

觀察

• 04a — Ctrl Timeseries：8 個 actuator 的控制信號。大致平滑，tick 邊界有微小跳變（對應 03b 的 qpos 不連續）。
• 04b — Ctrl Discontinuity Heatmap：tick 邊界 |Δctrl|，mean 0.012, max 0.089。比 qpos 的跳變更明顯但仍在可接受範圍。
• 04c — Ctrl Effort vs opt_steps：
  • ctrl_effort 和 opt_steps 相關性弱（r = 0.168）
  • ctrl_rate 和 opt_steps 弱負相關（r = −0.305）

對理論的修正

原本預期「optimizer 花越多步的 tick，控制信號應該越大/變化越快」。但數據顯示幾乎無關。解釋：opt_steps 高的原因是 reward landscape 複雜（contact），不是因為需要大控制量。控制信號的大小由任務幾何決定，而非優化難度。

───

5. 3D 空間軌跡（05 系列）

設計動機

在 joint space 看到的都是抽象數字。想在 Cartesian space 看 fingertip 的實際運動路徑，特別是 receding horizon 的 plan vs execute 差異。

觀察

• 05a — Receding Horizon 視覺化：3 個 representative sites（L-wrist, L-index, R-index），兩個視角。每個 tick 規劃 160 步（虛線），只執行前 40 步（實線, 25%）。虛線的尾端散開——離當前狀態越遠的規劃越不可靠，符合 receding horizon 只信任近期計劃的設計理念。
• 05b — Per-Tick Site Displacement：executed portion 的 per-site 位移。Site 1（L-thumb）幾乎不動（位移 ~0），wrist sites 位移最大。左右手活動量不對稱。
• 05c — Bimanual Paths：左右手平均軌跡 + gap detection。Inter-tick gaps 非常小（~1-2mm），幾乎連續——receding horizon 的執行段銜接良好。

對理論的驗證

Receding horizon 的核心假設（「只信近期」）在 3D 空間中得到直觀驗證：遠端規劃散開、近端執行緊密。同時確認了 tick 間的空間連續性，與 03b 的 joint-space 結果一致。

───

6. 跨 Rollout / 跨任務比較（06 系列）

設計動機

單一成功 rollout 看不出 SPIDER 的弱點。需要跨 rollout（同任務不同隨機種子）和跨任務（不同操作難度）的比較，才能找到 contact transition 的影響。

觀察

• 06a — Reward Across Rollouts：
  • p36-tea：rollout 0 在所有 17 個 tick 都是 outlier（reward 顯著低於 mean − 1σ），其餘 4 個 rollout 一致
  • p44-dog / p52-instrument：rollout 間一致性高，幾乎無 outlier
• 06b — Optimizer Effort Profile：
  • p36-tea 的 peak opt_steps 在 tick 9 附近（contact transition）
  • p44-dog
[2026/4/5 下午 04:07] 怪異專家: 的 peak 在 tick 0（開局就接觸物體）
  • p52-instrument 的 peak 在 tick 4
• 06c — Cross-Task Difficulty：
  • 難度排名：p36-tea（mean opt_steps = 6.8）> p52-instrument（4.5）> p44-dog（1.7）
  • p36-tea 顯著更難，因為 tea 任務涉及複雜的 bimanual 協調 + contact transition
• 06d — Reward Variance：
[2026/4/5 下午 04:07] 怪異專家: • Reward variance 的 peak 位置可作為 contact instability indicator
  • p36-tea 的 variance 遠高於其他兩個任務

對理論的驗證

Phase 3 預測的「contact boundary 不連續性」在跨 rollout 比較中得到量化驗證：

1. Contact transition tick 的 opt_steps spike 符合「feasible set 收縮」的預測
2. p36-tea rollout 0 整體 outlier 可能是初始 noise seed 落入了 homotopy path 斷裂的區域
3. 任務越複雜（contact modes 越多），MPPI 的穩定性越差——符合「O(2^{nm}) contact modes vs 1024 samples」的覆蓋率問題

───

討論收穫

Roy 從視覺化中學到的核心認知

1. Receding Horizon 的直觀意義：規劃 1.6s 只執行 0.4s，不是浪費——是用遠端計劃提供梯度方向，近端執行提供精確控制。3D 圖中虛線散開、實線緊密，完美體現了這個設計。
2. 不同 DOF 的行為差異是理所當然的：同時優化不等於同等重要。Wrist 負責大範圍移動，fingers 負責精細接觸，物體 DOF 被 contact force 主導。qpos >99.7% 的 cost share 說明 SPIDER 的策略是「先到位，再談力」。
3. 成功案例的表象會騙人：單一成功 rollout 看起來一切正常（reward 收斂、軌跡平滑、tick 連續）。只有跨 rollout 比較才能暴露 MPPI 在 contact transition 附近的脆弱性。
4. 四個關鍵診斷指標：
  • opt_steps — 優化收斂速度（越高 = 越難）
  • Reward 水準 — 追蹤品質（越負 = 越差）
  • Tick 邊界連續性 — receding horizon 銜接品質
  • 跨 rollout reward variance — 穩定性 / contact instability
5. Ctrl effort ≠ 優化難度：控制信號大小由任務幾何決定，opt_steps 多寡由 reward landscape 複雜度決定。兩者弱相關是合理的——它們回答不同問題。

───

Phase 4 完成條件

• [x] 至少一個自己寫的視覺化，能對應到論文的某個 figure（06 系列的跨任務比較對應論文 Table 1 的 success rate 排名）
• [x] 如果結果不同，能解釋為什麼（我們看的是 optimizer effort 而非 success rate，但難度排名一致）