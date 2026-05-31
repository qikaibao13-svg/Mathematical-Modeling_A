# 任务四脚本输入输出说明

## 脚本定位

`python -m src.task4.health_management_report` 是基于任务三输出的健康管理报告生成脚本。它不重新训练寿命模型，也不读取其他任务的结果文件，而是把任务三已经得到的附件2阶段、RUL、区间和误差证据转换为工程可读的规则分级预警与健康管理报告。

## 输入文件

| 文件 | 用途 |
|---|---|
| `outputs_sample/task3/attachment2_predictions.csv` | 读取任务三对附件2的 baseline 参照、残差修正量、最终 RUL、最终等效寿命坐标和阶段判断 |
| `outputs_sample/task3/attachment1_truncation_experiment.csv` | 读取任务三截断实验误差，用于给出 RUL 严重阈值和模型误差证据 |
| `outputs_sample/task3/uncertainty_interval_summary.csv` | 读取任务三整理后的 baseline 参照区间和最终保守区间 |

## 输出文件

| 文件 | 含义 |
|---|---|
| `warning_rules.csv` | 四级预警规则、触发条件、阈值来源和工程措施 |
| `warning_result.csv` / `warning_result.json` | 附件2当前健康状态、RUL、触发规则、预警等级和建议措施 |
| `health_management_report.md` | 面向工程决策的完整健康管理报告 |
| `paper_text.md` | 可迁移到论文正文的任务四表述 |
| `figures/life_rul_panel.png` | 等效寿命位置、RUL 区间与规则阈值图 |
| `figures/warning_level.png` | 当前预警等级图 |
| `figures/risk_breakdown.png` | 规则触发证据图 |

## 当前附件2结论

- 当前健康状态：退化期末段，接近衰退边界
- 最终等效 RUL：1045.93 天
- 保守区间：1017.12-1179.48 天
- 当前预警等级：2级警戒（黄色预警）
- 当前触发规则：R1_stage;R1_boundary;R2_boundary;U1_interval
