# 任务三  迁移残差修正模型结构

## 总体形式

```text
RUL_final = RUL_baseline + Delta_RUL_pred
```

其中 `RUL_baseline` 固定来自任务一飞轮本体模型，不被篡改。迁移模块只输出 baseline 残差修正量。

## 编码层

继续复用任务二 `BiLSTM-Attention` 的思想作为源域特征提取器基础：源域轴承和目标域飞轮都先进入退化语义特征空间，再通过 BiLSTM/Attention 表示退化上下文。

## 域适配层

保留直接迁移参考 DANN 框架：

- Gradient Reversal
- Domain Discriminator
- moment alignment

直接迁移参考已验证 DANN 表示对齐有效。本轮不让 DANN 直接输出绝对 RUL，而是作为残差修正头的跨域表示基础。

## 输出层

输出从 `pred_RUL_days` 改为：

```text
Delta_RUL_pred
```

辅助输入包括：

- baseline_RUL_days
- baseline_stage 编码
- 当前 HI_flywheel
- 主源域/辅助源域模板相似度权重

## 源域权重

| bearing_id | role      | dtw_to_flywheel | raw_similarity | prior_weight | source_weight |
| ---------- | --------- | --------------- | -------------- | ------------ | ------------- |
| Bearing3_1 | main      | 0.0194          | 51.5222        | 2.0000       | 0.2610        |
| Bearing2_5 | auxiliary | 0.0043          | 230.2512       | 1.0000       | 0.2108        |
| Bearing1_1 | auxiliary | 0.0041          | 244.7095       | 1.0000       | 0.2240        |
| Bearing3_5 | auxiliary | 0.0030          | 332.4016       | 1.0000       | 0.3043        |
