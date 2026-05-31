# Reaction Wheel Health Management Modeling

This repository contains the current implementation for reaction wheel degradation assessment, source-domain bearing modeling, transfer residual correction, and engineering health-management reporting.

## Contents

```text
src/            Current task scripts
docs/           Project and data notes
examples/       Representative figures
outputs_sample/ Compact example outputs
data/raw/       Raw data placeholder
```

## Task Chain

1. Task 1 builds the reaction wheel baseline and maps Attachment 2 to the equivalent life coordinate.
2. Task 2 models bearing degradation features and source-domain health indicators.
3. Task 3 applies baseline-constrained transfer residual correction.
4. Task 4 is based on Task 3 outputs only and generates a rule-based threshold warning report.

## Quick Run

Task 4 can be run directly from the included sample outputs:

```bash
python -m src.task4.health_management_report
```

For the full pipeline, place raw datasets as described in `docs/data_requirements.md`, then run:

```bash
python -m src.task1.reaction_wheel_baseline
python -m src.task2.bearing_source_model
python -m src.task3.transfer_feature_alignment
python -m src.task3.transfer_residual_correction
python -m src.task4.health_management_report
```

## Result Snapshot

- Attachment 2 current state: late degradation stage, near the decline boundary.
- Transfer-corrected equivalent RUL: about 1045.93 days.
- Conservative interval: about 1017.12-1179.48 days.
- Warning level: Level 2 yellow warning.
- Warning method: rule-based threshold logic, not manually weighted risk scoring.
