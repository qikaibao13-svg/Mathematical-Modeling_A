# 任务三直接迁移参考失败原因复核

## 1. 直接预测绝对 RUL 过难

直接迁移参考 DANN 模型直接输出 `pred_RUL_days`。这要求迁移模型同时学习轴承到飞轮的退化表示、飞轮 3500 天寿命尺度以及附件2当前寿命位置，任务过重。

## 2. 域对齐没有自动转化为 RUL 优势

直接迁移参考域对齐指标如下：

| metric                                      | value   |
| ------------------------------------------- | ------- |
| centroid_distance_source_attachment1_before | 2.4071  |
| centroid_distance_source_attachment1_after  | 0.8956  |
| centroid_distance_source_attachment2_after  | 0.9106  |
| embedding_dim                               | 96.0000 |

源域与附件1表示距离明显降低，说明域适配成功；但 RUL 误差没有全面降低，说明“表示接近”不等于“绝对寿命尺度预测更准”。

## 3. 误差放大点

直接迁移参考最大 transfer 误差出现在 60% 截断点，误差为 253.08 天。只有 1 / 4 个截断点优于 baseline。

| fraction | observed_day | true_RUL_days | baseline_RUL_days | baseline_abs_error_days | transfer_RUL_days | transfer_abs_error_days |
| -------- | ------------ | ------------- | ----------------- | ----------------------- | ----------------- | ----------------------- |
| 0.6000   | 2100.0000    | 1400.0000     | 1348.9757         | 51.0243                 | 1146.9180         | 253.0820                |
| 0.7000   | 2440.0000    | 1060.0000     | 1010.4192         | 49.5808                 | 979.8040          | 80.1960                 |
| 0.8000   | 2800.0000    | 700.0000      | 652.2773          | 47.7227                 | 656.6814          | 43.3186                 |
| 0.9000   | 3160.0000    | 340.0000      | 294.2966          | 45.7034                 | 121.8526          | 218.1474                |

## 4. baseline 的优势

任务一 baseline 直接由飞轮附件1全寿命趋势拟合得到，天然掌握飞轮本体时间尺度；其主要问题是存在约 45-51 天的系统性保守偏差。因此  不再让迁移模型替代 baseline，而是学习：

```text
Delta_RUL = RUL_true - RUL_baseline
RUL_final = RUL_baseline + Delta_RUL_pred
```
