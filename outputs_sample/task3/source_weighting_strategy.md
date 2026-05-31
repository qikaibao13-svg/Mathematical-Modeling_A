# 源域加权策略

本轮残差修正不把四个源域等权混合。主源域 `Bearing3_1` 保持最高先验权重，辅助源域 `Bearing2_5/Bearing1_1/Bearing3_5` 根据与飞轮附件1 HI 模板的 DTW 相似度加权。

| bearing_id | role      | dtw_to_flywheel | raw_similarity | prior_weight | source_weight |
| ---------- | --------- | --------------- | -------------- | ------------ | ------------- |
| Bearing3_1 | main      | 0.0194          | 51.5222        | 2.0000       | 0.2610        |
| Bearing2_5 | auxiliary | 0.0043          | 230.2512       | 1.0000       | 0.2108        |
| Bearing1_1 | auxiliary | 0.0041          | 244.7095       | 1.0000       | 0.2240        |
| Bearing3_5 | auxiliary | 0.0030          | 332.4016       | 1.0000       | 0.3043        |

设计理由：

1. `Bearing3_1` 生命周期完整、阶段边界清晰，是主源域；
2. 辅助源域只提供退化形态补充，避免明显负迁移；
3. 残差修正头只学习 baseline 系统偏差，不直接替代 baseline。
