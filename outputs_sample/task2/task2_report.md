# 任务二最新版正式执行报告

## 数据与流程

真实读取 XJTU-SY 数据目录，确认共有 15 个轴承、9216 个分钟级 CSV。分钟级特征表来自已完成的真实原始 CSV 特征提取结果，本轮重新执行 EWM 筛选、HI 构造和正式四模型比较。

## 特征筛选

特征筛选正式复用 *Remaining useful life prediction of rolling bearings via wavelet feature fusion and LSTM networks* 的 `Correlation + Monotonicity + Robustness + EWM` 思想，没有使用五维人工评分。

EWM 权重：

| criterion    | weight |
| ------------ | ------ |
| Correlation  | 0.1745 |
| Monotonicity | 0.5147 |
| Robustness   | 0.3107 |

最终核心特征：

| feature                 | Correlation | Monotonicity | Robustness | J_score | HI_weight |
| ----------------------- | ----------- | ------------ | ---------- | ------- | --------- |
| fused_rms_vector        | 0.7323      | 0.6999       | 0.9832     | 0.7936  | 0.1007    |
| fused_rms_mean          | 0.7328      | 0.6974       | 0.9831     | 0.7924  | 0.1005    |
| fused_std_mean          | 0.7326      | 0.6944       | 0.9835     | 0.7909  | 0.1003    |
| fused_envelope_rms_mean | 0.7325      | 0.6944       | 0.9835     | 0.7909  | 0.1003    |
| fused_rms_max           | 0.7270      | 0.6926       | 0.9824     | 0.7887  | 0.1001    |
| fused_std_max           | 0.7270      | 0.6916       | 0.9826     | 0.7882  | 0.1000    |
| fused_envelope_rms_max  | 0.7270      | 0.6916       | 0.9826     | 0.7882  | 0.1000    |
| fused_total_energy_sum  | 0.6704      | 0.7001       | 0.9851     | 0.7835  | 0.0994    |
| fused_total_energy_mean | 0.6704      | 0.7001       | 0.9851     | 0.7835  | 0.0994    |
| fused_variance_mean     | 0.6704      | 0.6968       | 0.9852     | 0.7818  | 0.0992    |

## HI_bearing 与阶段边界

`HI_bearing` 由核心特征按 `J_score` 归一化权重融合，并经过轻度平滑和 cumulative max 修正。

主源域/辅助源域三阶段边界：

| bearing_id | tau1_life_ratio | tau2_life_ratio | method                               |
| ---------- | --------------- | --------------- | ------------------------------------ |
| Bearing1_1 | 0.6066          | 0.6885          | three-piece linear SSE on HI_bearing |
| Bearing2_5 | 0.4349          | 0.9053          | three-piece linear SSE on HI_bearing |
| Bearing3_1 | 0.8203          | 0.9200          | three-piece linear SSE on HI_bearing |
| Bearing3_5 | 0.4159          | 0.8319          | three-piece linear SSE on HI_bearing |

## 正式四模型统一比较

正式主比较模型为 PF-based、EKF-based、Random Forest、BiLSTM-Attention。四个模型均在同一测试轴承、同一 inspection points、同一全寿命伪在线截断规则下预测 `RUL_ratio`。

| model            | MAE    | MSE    | RMSE   | Score  | seq_len | rank_by_score | rank_by_rmse |
| ---------------- | ------ | ------ | ------ | ------ | ------- | ------------- | ------------ |
| Random Forest    | 0.2948 | 0.1227 | 0.3502 | 0.0799 |         | 1             | 2            |
| BiLSTM-Attention | 0.2342 | 0.0743 | 0.2725 | 0.0673 | 16.0000 | 2             | 1            |
| EKF-based        | 0.3884 | 0.1998 | 0.4470 | 0.0184 |         | 3             | 4            |
| PF-based         | 0.3523 | 0.1519 | 0.3898 | 0.0153 |         | 4             | 3            |

正式最优模型：`Random Forest`。

说明：按正式 Score 口径，Random Forest 排名第一；按 MAE/MSE/RMSE 误差口径，BiLSTM-Attention 最低。因此本轮结论应写为“Random Forest 在当前 Score 规则下综合最优，BiLSTM-Attention 的点误差拟合能力最好”。

## Hybrid 文献复现参考

Hybrid_RVM_exponential_Frechet 只作为文献复现参考，不进入正式主比较表。

| model                  | MAE    | MSE    | RMSE   | Score  |
| ---------------------- | ------ | ------ | ------ | ------ |
| Hybrid_RVM_exp_Frechet | 0.2513 | 0.0807 | 0.2841 | 0.0776 |

## 最新方案符合性检查

1. 未使用五维人工评分。
2. 未使用 afterSPT、deg_RUL_ratio、退化阶段增强实验。
3. 正式模型统一预测 `RUL_ratio`。
4. 模型输入不包含 `life_ratio/tau1/deg_progress/deg_RUL_ratio` 等泄漏变量。
5. Hybrid 单独作为参考输出，不进入 `task2_main_model_comparison.csv`。
6. BiLSTM-Attention 为单任务 RUL 输出，并试验窗口长度 16/32/64。

## 图表

- `figures/task2_selected_feature_scores.png`
- `figures/task2_source_hi_stage_boundaries.png`
- `figures/task2_main_model_comparison.png`
