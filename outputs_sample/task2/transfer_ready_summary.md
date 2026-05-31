# 任务三可直接使用的源域接口

## selected_features

fused_rms_vector, fused_rms_mean, fused_std_mean, fused_envelope_rms_mean, fused_rms_max, fused_std_max, fused_envelope_rms_max, fused_total_energy_sum, fused_total_energy_mean, fused_variance_mean

## HI_bearing 与标签

- `task2_HI_bearing.csv`
- `task2_training_labels.csv`

## source_bearing_candidates

| condition  | bearing_id | fault_type | file_count | source_score | recommended_role |
| ---------- | ---------- | ---------- | ---------- | ------------ | ---------------- |
| 40Hz10kN   | Bearing3_1 | Outer race | 2538       | 0.9967       | primary_source   |
| 37.5Hz11kN | Bearing2_5 | Outer race | 339        | 0.9150       | auxiliary_source |
| 35Hz12kN   | Bearing1_1 | Outer race | 123        | 0.8115       | auxiliary_source |
| 40Hz10kN   | Bearing3_5 | Outer race | 114        | 0.8070       | auxiliary_source |
| 35Hz12kN   | Bearing1_2 | Outer race | 161        | 0.7698       | candidate        |
| 37.5Hz11kN | Bearing2_2 | Outer race | 161        | 0.7698       | candidate        |
| 35Hz12kN   | Bearing1_3 | Outer race | 158        | 0.7690       | candidate        |
| 37.5Hz11kN | Bearing2_4 | Outer race | 42         | 0.7101       | candidate        |

## stage_boundaries

`task2_stage_boundaries.csv`

## normalized_degradation_template

`task2_normalized_degradation_template.csv`

## best_source_model

`Random Forest`，对应正式比较见 `task2_main_model_comparison.csv`。

## 注意

轴承分钟级寿命不直接换算为飞轮天数；任务三应迁移归一化退化模板、阶段结构和源模型知识。
