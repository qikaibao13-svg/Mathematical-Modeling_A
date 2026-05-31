# Task 4 Notes

Task 4 reads Task 3 outputs and generates the warning result, warning rules, figures, and health-management report for Attachment 2.

It has no direct Task 1 dependency. The baseline values used in Task 4 are the baseline reference columns already carried by Task 3 output files, not separate Task 1 result files.

The formal warning mechanism is rule-based threshold triggering. It follows the common PHM pattern of checking a health-stage or threshold crossing first, then reporting RUL and engineering actions. It does not use manually assigned weights as the formal decision rule.

## Inputs

- `outputs_sample/task3/attachment2_predictions.csv`
- `outputs_sample/task3/uncertainty_interval_summary.csv`
- `outputs_sample/task3/attachment1_truncation_experiment.csv`

## Outputs

- `outputs_sample/task4/warning_result.csv`
- `outputs_sample/task4/warning_result.json`
- `outputs_sample/task4/warning_rules.csv`
- `outputs_sample/task4/health_management_report.md`
- `outputs_sample/task4/paper_text.md`
- `outputs_sample/task4/input_output_description.md`
- `outputs_sample/task4/figures/life_rul_panel.png`
- `outputs_sample/task4/figures/warning_level.png`
- `outputs_sample/task4/figures/risk_breakdown.png`

## Current Result

- Current state: late degradation stage, near the decline boundary.
- Warning level: Level 2 yellow warning.
- Triggered rules: `R1_stage`, `R1_boundary`, `R2_boundary`, and `U1_interval`.
- Rule severity index: 66.67/100, derived only from the final warning level for display.
- Transfer-corrected equivalent RUL: 1045.93 days.
- Conservative interval: 1017.12-1179.48 days.
