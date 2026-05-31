# 任务三 ：baseline + 迁移残差修正执行报告

## 方法定位

 不再让迁移模型直接替代任务一 baseline，而是学习 baseline 的系统性偏差：

```text
Delta_RUL_label = RUL_true - RUL_baseline
RUL_final = RUL_baseline + Delta_RUL_pred
```

## 失败原因复核

直接迁移参考 DANN 域对齐成功，但直接绝对 RUL 输出没有全面超过 baseline。原因是 baseline 已掌握飞轮本体时间尺度，而直接迁移模型要同时学习跨域表示和飞轮寿命尺度，容易过度修正。

## 附件1截断实验

| fraction | observed_day | true_RUL_days | baseline_RUL_days | Delta_RUL_pred | RUL_final | baseline_abs_error_days | final_abs_error_days | baseline_improvement_days |
| -------- | ------------ | ------------- | ----------------- | -------------- | --------- | ----------------------- | -------------------- | ------------------------- |
| 0.6000   | 2100.0000    | 1400.0000     | 1347.1337         | 45.0181        | 1392.1518 | 52.8663                 | 7.8482               | 45.0181                   |
| 0.7000   | 2440.0000    | 1060.0000     | 1013.7201         | 46.5241        | 1060.2442 | 46.2799                 | 0.2442               | 46.0357                   |
| 0.8000   | 2800.0000    | 700.0000      | 656.9910          | 45.8606        | 702.8515  | 43.0090                 | 2.8515               | 40.1575                   |
| 0.9000   | 3140.0000    | 360.0000      | 318.4873          | 43.4489        | 361.9362  | 41.5127                 | 1.9362               | 39.5766                   |

## 附件2最终预测

| observed_day | baseline_RUL_days | baseline_stage | baseline_HI | equivalent_life_day | Delta_RUL_pred | RUL_final | transfer_corrected_stage | equivalent_life_day_final | selected_variant |
| ------------ | ----------------- | -------------- | ----------- | ------------------- | -------------- | --------- | ------------------------ | ------------------------- | ---------------- |
| 1800.0000    | 1000.4869         | 退化期            | 0.6001      | 2499.5131           | 45.4412        | 1045.9282 | 退化期                      | 2454.0718                 | hi_linear_shrink |

## 调参摘要

| variant                   | seq_len | dann_weight | moment | aux_input | source_weighting | loocv_MAE | loocv_RMSE | improvement_vs_baseline_MAE | mean_abs_delta_pred |
| ------------------------- | ------- | ----------- | ------ | --------- | ---------------- | --------- | ---------- | --------------------------- | ------------------- |
| no_correction             | 0       | 0.0000      | False  | False     | False            | 45.9170   | 46.1242    | 0.0000                      | 0.0000              |
| mean                      | 16      | 0.0000      | False  | True      | False            | 4.8748    | 5.8225     | 41.0422                     | 45.9170             |
| late_weighted             | 16      | 0.3500      | False  | True      | True             | 4.8451    | 5.7478     | 41.0719                     | 45.5838             |
| hi_linear_shrink          | 16      | 0.3500      | True   | True      | True             | 3.2200    | 4.2876     | 42.6970                     | 45.2129             |
| dann_residual_constrained | 16      | 0.3500      | True   | True      | True             | 4.8370    | 5.7330     | 41.0800                     | 45.4920             |

## 结论

 在附件1截断实验中显著降低 baseline 系统性保守偏差，并避免直接迁移参考模型的大幅过度修正。附件2上， 给出温和正残差修正，使 RUL 从 baseline 的 1000.49 天调整为 1045.93 天。
