# 任务三  损失函数与训练策略

## 残差标签

```text
Delta_RUL_label = RUL_true - RUL_baseline
```

## 损失函数

```text
L = L_residual + lambda1 * L_DA + lambda2 * L_moment + lambda3 * L_amplitude
```

其中：

- `L_residual = MSE(Delta_RUL_label, Delta_RUL_pred)`
- `L_DA` 来自 DANN 域判别器
- `L_moment` 约束源/目标表示一阶矩接近
- `L_amplitude` 限制修正幅度，避免过度修正

## 后期样本加权

截断点越靠近寿命末端，残差损失权重越高：

```text
beta = 1 + 2 * life_ratio
```

## 有限调参结果

| variant                   | seq_len | dann_weight | moment | aux_input | source_weighting | loocv_MAE | loocv_RMSE | improvement_vs_baseline_MAE | mean_abs_delta_pred |
| ------------------------- | ------- | ----------- | ------ | --------- | ---------------- | --------- | ---------- | --------------------------- | ------------------- |
| no_correction             | 0       | 0.0000      | False  | False     | False            | 45.9170   | 46.1242    | 0.0000                      | 0.0000              |
| mean                      | 16      | 0.0000      | False  | True      | False            | 4.8748    | 5.8225     | 41.0422                     | 45.9170             |
| late_weighted             | 16      | 0.3500      | False  | True      | True             | 4.8451    | 5.7478     | 41.0719                     | 45.5838             |
| hi_linear_shrink          | 16      | 0.3500      | True   | True      | True             | 3.2200    | 4.2876     | 42.6970                     | 45.2129             |
| dann_residual_constrained | 16      | 0.3500      | True   | True      | True             | 4.8370    | 5.7330     | 41.0800                     | 45.4920             |

最终采用 `hi_linear_shrink`。该方案使用 leave-one-cutoff 方式评估附件1截断点，避免直接用同一截断点标签训练并评价同一截断点。
