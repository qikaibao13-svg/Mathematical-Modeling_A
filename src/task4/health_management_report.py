from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(Path(os.environ.get("TEMP", str(PROJECT_ROOT))) / "mm_contest_mplconfig_task4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUTS = PROJECT_ROOT / "outputs"
SAMPLES = PROJECT_ROOT / "outputs_sample"


def existing_output_dir(name: str) -> Path:
    generated = OUTPUTS / name
    if generated.exists():
        return generated
    return SAMPLES / name


TASK3 = existing_output_dir("task3")
TASK3_INTERVAL = TASK3
OUT = OUTPUTS / "task4"
FIG = OUT / "figures"

for path in [OUT, FIG]:
    path.mkdir(parents=True, exist_ok=True)

PREDICTION_FILE = TASK3 / "attachment2_predictions.csv"
TRUNCATION_FILE = TASK3 / "attachment1_truncation_experiment.csv"
INTERVAL_FILE = TASK3_INTERVAL / "uncertainty_interval_summary.csv"

TOTAL_LIFE = 3500.0
DECLINE_BOUNDARY_DAY = 2500.0

COLORS = {
    "blue": "#2F6FBB",
    "green": "#4C956C",
    "yellow": "#F2C94C",
    "orange": "#E6862D",
    "red": "#C44E52",
    "gray": "#6E7781",
    "dark": "#222222",
    "light_blue": "#EAF3FF",
    "light_green": "#EAF7EF",
    "light_yellow": "#FFF8D8",
    "light_red": "#FDECEC",
}

FONT_CANDIDATES = [
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
]
FONT_PATH = next((p for p in FONT_CANDIDATES if p.exists()), None)
if FONT_PATH is not None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": FONT_FAMILY,
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.titleweight": "bold",
    }
)


def load_task3_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Task 4 is a report layer; all decision inputs are read from Task 3 outputs."""
    prediction = pd.read_csv(PREDICTION_FILE).iloc[0]
    truncation = pd.read_csv(TRUNCATION_FILE)
    interval = pd.read_csv(INTERVAL_FILE)
    return prediction, truncation, interval


def get_interval(interval: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "baseline":
        mask = interval["method"].astype(str).str.contains("baseline_conservative", regex=False)
    else:
        mask = interval["method"].astype(str).str.contains("conservative_dual_path_plus_residual_rmse", regex=False)
    if not mask.any():
        raise ValueError(f"Missing {kind} interval row in {INTERVAL_FILE}")
    return interval[mask].iloc[0]


def derive_thresholds(truncation: pd.DataFrame) -> dict[str, float]:
    """Derive rule thresholds from Task 3 stage structure and truncation evidence."""
    boundary_rul = TOTAL_LIFE - DECLINE_BOUNDARY_DAY
    decline_rows = truncation[truncation["baseline_stage"].astype(str).eq("衰退期")]
    if decline_rows.empty:
        severe_rul = 0.10 * TOTAL_LIFE
    else:
        severe_rul = float(decline_rows["true_RUL_days"].astype(float).min())
    final_mae = float(truncation["final_abs_error_days"].astype(float).mean())
    baseline_mae = float(truncation["baseline_abs_error_days"].astype(float).mean())
    return {
        "decline_boundary_day": DECLINE_BOUNDARY_DAY,
        "boundary_rul_days": boundary_rul,
        "attention_boundary_window_days": 0.30 * boundary_rul,
        "warning_boundary_window_days": 0.10 * boundary_rul,
        "attention_rul_low_days": 1.50 * boundary_rul,
        "warning_rul_low_days": boundary_rul,
        "severe_rul_low_days": severe_rul,
        "attention_interval_ratio": 0.15,
        "warning_interval_ratio": 0.25,
        "severe_interval_ratio": 0.50,
        "task3_final_mae_days": final_mae,
        "task3_baseline_mae_days": baseline_mae,
    }


def build_warning_rules(thresholds: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "rule_id": "R0",
            "level": 0,
            "warning_name": "正常",
            "color": "蓝色",
            "trigger_condition": "未触发 R1-R3 规则",
            "threshold_source": "PHM control-limit logic: no stage/RUL threshold crossing",
            "engineering_action": "维持常规遥测监测和周期性趋势复核。",
        },
        {
            "rule_id": "R1_stage",
            "level": 1,
            "warning_name": "关注",
            "color": "绿色",
            "trigger_condition": "transfer_corrected_stage 为退化期",
            "threshold_source": "Task 3 stage label",
            "engineering_action": "提高趋势复核频率，关注电流、温度和健康指标的同步变化。",
        },
        {
            "rule_id": "R1_boundary",
            "level": 1,
            "warning_name": "关注",
            "color": "绿色",
            "trigger_condition": f"final 等效寿命距衰退边界 <= {thresholds['attention_boundary_window_days']:.0f} 天",
            "threshold_source": "30% of the RUL at the decline boundary, used as an early management window",
            "engineering_action": "缩短复核间隔，观察是否持续向衰退边界靠近。",
        },
        {
            "rule_id": "R2_boundary",
            "level": 2,
            "warning_name": "警戒",
            "color": "黄色",
            "trigger_condition": f"final 等效寿命距衰退边界 <= {thresholds['warning_boundary_window_days']:.0f} 天",
            "threshold_source": "10% of the RUL at the decline boundary, following threshold/control-limit warning logic",
            "engineering_action": "进入加密监测状态，准备姿态控制冗余策略和任务窗口调整预案。",
        },
        {
            "rule_id": "R2_rul",
            "level": 2,
            "warning_name": "警戒",
            "color": "黄色",
            "trigger_condition": f"RUL 保守区间下界 <= {thresholds['warning_rul_low_days']:.0f} 天",
            "threshold_source": "Task 3 decline boundary corresponds to about 1000 days RUL",
            "engineering_action": "将飞轮纳入寿命窗口管理，限制非必要高负载暴露。",
        },
        {
            "rule_id": "R3_stage",
            "level": 3,
            "warning_name": "严重",
            "color": "红色",
            "trigger_condition": "transfer_corrected_stage 为衰退期，或 final 等效寿命越过衰退边界",
            "threshold_source": "Task 3 stage boundary and PHM failure-threshold logic",
            "engineering_action": "启动维护、替换或冗余执行机构接管预案。",
        },
        {
            "rule_id": "R3_rul",
            "level": 3,
            "warning_name": "严重",
            "color": "红色",
            "trigger_condition": f"RUL 保守区间下界 <= {thresholds['severe_rul_low_days']:.0f} 天",
            "threshold_source": "Minimum true RUL among decline-stage truncation samples in Task 3",
            "engineering_action": "限制高负载工况，启动冗余接管或维护预案。",
        },
        {
            "rule_id": "U1_interval",
            "level": 1,
            "warning_name": "关注",
            "color": "绿色",
            "trigger_condition": f"RUL 区间宽度 / RUL 点估计 >= {thresholds['attention_interval_ratio']:.2f}",
            "threshold_source": "Task 3 conservative interval width; used only as uncertainty attention flag",
            "engineering_action": "提示区间不确定性偏大，需随新增数据滚动更新。",
        },
        {
            "rule_id": "U2_interval",
            "level": 2,
            "warning_name": "警戒",
            "color": "黄色",
            "trigger_condition": f"RUL 区间宽度 / RUL 点估计 >= {thresholds['warning_interval_ratio']:.2f}",
            "threshold_source": "Task 3 conservative interval width; used only as uncertainty attention flag",
            "engineering_action": "在行动建议中采用更保守的任务窗口。",
        },
    ]
    return pd.DataFrame(rows)


def evaluate_warning(
    prediction: pd.Series,
    interval: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, float | int | str]:
    baseline_interval = get_interval(interval, "baseline")
    final_interval = get_interval(interval, "final")

    baseline_life = float(prediction["equivalent_life_day"])
    final_life = float(prediction["equivalent_life_day_final"])
    final_rul = float(prediction["RUL_final"])
    rul_low = float(final_interval["RUL_low"])
    rul_high = float(final_interval["RUL_high"])
    interval_width = rul_high - rul_low
    interval_ratio = interval_width / max(final_rul, 1.0)
    distance_final = thresholds["decline_boundary_day"] - final_life
    distance_baseline = thresholds["decline_boundary_day"] - baseline_life

    triggered: list[tuple[str, int, str]] = []
    stage = str(prediction["transfer_corrected_stage"])
    if stage == "退化期":
        triggered.append(("R1_stage", 1, "任务三修正后阶段为退化期"))
    if distance_final <= thresholds["attention_boundary_window_days"]:
        triggered.append(("R1_boundary", 1, f"距衰退边界 {distance_final:.2f} 天，低于关注窗口"))
    if distance_final <= thresholds["warning_boundary_window_days"]:
        triggered.append(("R2_boundary", 2, f"距衰退边界 {distance_final:.2f} 天，低于警戒窗口"))
    if rul_low <= thresholds["warning_rul_low_days"]:
        triggered.append(("R2_rul", 2, f"RUL 下界 {rul_low:.2f} 天触及警戒阈值"))
    if stage == "衰退期" or final_life >= thresholds["decline_boundary_day"]:
        triggered.append(("R3_stage", 3, "已进入或越过衰退边界"))
    if rul_low <= thresholds["severe_rul_low_days"]:
        triggered.append(("R3_rul", 3, f"RUL 下界 {rul_low:.2f} 天触及严重阈值"))
    if interval_ratio >= thresholds["attention_interval_ratio"]:
        triggered.append(("U1_interval", 1, f"区间宽度比例 {interval_ratio:.3f} 触及不确定性关注阈值"))
    if interval_ratio >= thresholds["warning_interval_ratio"]:
        triggered.append(("U2_interval", 2, f"区间宽度比例 {interval_ratio:.3f} 触及不确定性警戒阈值"))

    if triggered:
        level = max(item[1] for item in triggered)
    else:
        level = 0

    names = {
        0: ("正常", "蓝色"),
        1: ("关注", "绿色"),
        2: ("警戒", "黄色"),
        3: ("严重", "红色"),
    }
    warning_name, warning_color = names[level]

    return {
        "warning_level": level,
        "warning_name": warning_name,
        "warning_color": warning_color,
        "triggered_rule_ids": ";".join(item[0] for item in triggered) if triggered else "R0",
        "triggered_rule_descriptions": "；".join(item[2] for item in triggered) if triggered else "未触发关注、警戒或严重规则",
        "rule_decision_basis": "采用最高触发等级作为当前预警等级，不使用人工加权风险分数。",
        "rule_severity_index": level / 3.0 * 100.0,
        "decline_boundary_day": thresholds["decline_boundary_day"],
        "distance_to_decline_boundary_days": distance_final,
        "baseline_distance_to_decline_boundary_days": distance_baseline,
        "boundary_attention_window_days": thresholds["attention_boundary_window_days"],
        "boundary_warning_window_days": thresholds["warning_boundary_window_days"],
        "RUL_low_conservative": rul_low,
        "RUL_high_conservative": rul_high,
        "RUL_interval_width": interval_width,
        "RUL_interval_ratio": interval_ratio,
        "RUL_warning_threshold_days": thresholds["warning_rul_low_days"],
        "RUL_severe_threshold_days": thresholds["severe_rul_low_days"],
        "uncertainty_attention_threshold": thresholds["attention_interval_ratio"],
        "uncertainty_warning_threshold": thresholds["warning_interval_ratio"],
        "baseline_RUL_low_conservative": float(baseline_interval["RUL_low"]),
        "baseline_RUL_high_conservative": float(baseline_interval["RUL_high"]),
    }


def current_state_text(prediction: pd.Series, decision: dict[str, float | int | str]) -> str:
    stage = str(prediction["transfer_corrected_stage"])
    distance = float(decision["distance_to_decline_boundary_days"])
    if stage == "衰退期":
        return "衰退期，高风险运行状态"
    if stage == "退化期" and distance <= float(decision["boundary_warning_window_days"]):
        return "退化期末段，接近衰退边界"
    if stage == "退化期":
        return "退化期，需持续跟踪"
    return "健康期或轻微退化状态"


def build_warning_result(prediction: pd.Series, decision: dict[str, float | int | str]) -> pd.DataFrame:
    state = current_state_text(prediction, decision)
    reason = (
        f"任务三附件2结果显示，baseline 参照 RUL 为 {float(prediction['baseline_RUL_days']):.2f} 天，"
        f"迁移残差修正量为 {float(prediction['Delta_RUL_pred']):.2f} 天，"
        f"最终等效 RUL 为 {float(prediction['RUL_final']):.2f} 天；"
        f"final 等效寿命坐标为 {float(prediction['equivalent_life_day_final']):.2f} 天，"
        f"距离衰退边界约 {float(decision['distance_to_decline_boundary_days']):.2f} 天。"
        f"当前触发规则：{decision['triggered_rule_descriptions']}。"
    )
    row = {
        "object": "附件2飞轮",
        "input_source": "task3_outputs_only",
        "warning_method": "rule_based_threshold_warning",
        "observed_day": float(prediction["observed_day"]),
        "baseline_stage": str(prediction["baseline_stage"]),
        "transfer_corrected_stage": str(prediction["transfer_corrected_stage"]),
        "current_HI": float(prediction["baseline_HI"]),
        "baseline_equivalent_life_day": float(prediction["equivalent_life_day"]),
        "final_equivalent_life_day": float(prediction["equivalent_life_day_final"]),
        "decline_boundary_day": float(decision["decline_boundary_day"]),
        "distance_to_decline_boundary_days": float(decision["distance_to_decline_boundary_days"]),
        "baseline_distance_to_decline_boundary_days": float(decision["baseline_distance_to_decline_boundary_days"]),
        "baseline_RUL_days": float(prediction["baseline_RUL_days"]),
        "Delta_RUL_pred_days": float(prediction["Delta_RUL_pred"]),
        "RUL_final_days": float(prediction["RUL_final"]),
        "baseline_RUL_low_conservative": float(decision["baseline_RUL_low_conservative"]),
        "baseline_RUL_high_conservative": float(decision["baseline_RUL_high_conservative"]),
        "RUL_low_conservative": float(decision["RUL_low_conservative"]),
        "RUL_high_conservative": float(decision["RUL_high_conservative"]),
        "RUL_interval_width": float(decision["RUL_interval_width"]),
        "RUL_interval_ratio": float(decision["RUL_interval_ratio"]),
        "boundary_attention_window_days": float(decision["boundary_attention_window_days"]),
        "boundary_warning_window_days": float(decision["boundary_warning_window_days"]),
        "RUL_warning_threshold_days": float(decision["RUL_warning_threshold_days"]),
        "RUL_severe_threshold_days": float(decision["RUL_severe_threshold_days"]),
        "current_state": state,
        "warning_level": int(decision["warning_level"]),
        "warning_name": str(decision["warning_name"]),
        "warning_color": str(decision["warning_color"]),
        "rule_severity_index": float(decision["rule_severity_index"]),
        "triggered_rule_ids": str(decision["triggered_rule_ids"]),
        "triggered_rule_descriptions": str(decision["triggered_rule_descriptions"]),
        "rule_decision_basis": str(decision["rule_decision_basis"]),
        "main_reason": reason,
        "recommended_action": "加密监测电流、温度与健康指标趋势；在高载荷姿态控制任务前准备冗余策略；若 final 等效寿命越过衰退边界或 RUL 区间下界快速下降，则升级预警。",
        "script_role": "读取任务三对附件2的最终预测与不确定性区间，生成规则分级预警和工程决策报告；不重新训练寿命模型。",
    }
    return pd.DataFrame([row])


def draw_warning_level(result: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 3.8))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blocks = [
        (0, "正常", COLORS["light_blue"], COLORS["blue"]),
        (1, "关注", COLORS["light_green"], COLORS["green"]),
        (2, "警戒", COLORS["light_yellow"], COLORS["yellow"]),
        (3, "严重", COLORS["light_red"], COLORS["red"]),
    ]
    current = int(result["warning_level"])
    for idx, label, face, edge in blocks:
        ax.barh(0.5, 0.92, left=idx + 0.04, height=0.38, color=face, edgecolor=edge, linewidth=1.6)
        ax.text(idx + 0.5, 0.50, f"{idx}级\n{label}", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["dark"])
    ax.scatter([current + 0.5], [0.83], s=220, color=COLORS["orange"], zorder=5)
    ax.text(current + 0.5, 0.94, f"附件2当前：{current}级{result['warning_name']}", ha="center", va="bottom", fontsize=12.5, color=COLORS["dark"], fontweight="bold")
    ax.text(2.0, 0.10, f"判定依据：最高触发规则；状态：{result['current_state']}", ha="center", fontsize=11)
    fig.savefig(FIG / "warning_level.png", bbox_inches="tight")
    plt.close(fig)


def draw_risk_breakdown(result: pd.Series) -> None:
    labels = ["阶段状态", "距衰退边界", "RUL下界", "区间宽度"]
    levels = [
        1 if str(result["transfer_corrected_stage"]) == "退化期" else 3 if str(result["transfer_corrected_stage"]) == "衰退期" else 0,
        2 if float(result["distance_to_decline_boundary_days"]) <= float(result["boundary_warning_window_days"]) else 1,
        2 if float(result["RUL_low_conservative"]) <= float(result["RUL_warning_threshold_days"]) else 0,
        2 if float(result["RUL_interval_ratio"]) >= 0.25 else 1 if float(result["RUL_interval_ratio"]) >= 0.15 else 0,
    ]
    colors = [COLORS["green"] if v == 1 else COLORS["yellow"] if v == 2 else COLORS["red"] if v == 3 else COLORS["blue"] for v in levels]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    y = np.arange(len(labels))
    ax.barh(y, levels, color=colors, alpha=0.88)
    texts = [
        str(result["transfer_corrected_stage"]),
        f"{float(result['distance_to_decline_boundary_days']):.1f} d <= {float(result['boundary_warning_window_days']):.0f} d",
        f"{float(result['RUL_low_conservative']):.1f} d / 阈值 {float(result['RUL_warning_threshold_days']):.0f} d",
        f"{float(result['RUL_interval_ratio']):.3f}",
    ]
    for i, (value, text) in enumerate(zip(levels, texts)):
        ax.text(value + 0.05, i, text, va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["正常", "关注", "警戒", "严重"])
    ax.set_xlabel("触发等级")
    ax.set_title("任务四规则触发证据")
    fig.savefig(FIG / "risk_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def draw_life_rul_panel(result: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    baseline_life = float(result["baseline_equivalent_life_day"])
    final_life = float(result["final_equivalent_life_day"])
    boundary = float(result["decline_boundary_day"])
    warning_left = boundary - float(result["boundary_warning_window_days"])
    ax.axvspan(0, warning_left, color=COLORS["light_blue"], label="常规/关注区")
    ax.axvspan(warning_left, boundary, color=COLORS["light_yellow"], label="警戒窗口")
    ax.axvspan(boundary, TOTAL_LIFE, color=COLORS["light_red"], label="衰退/严重区")
    ax.axvline(boundary, color=COLORS["gray"], linewidth=1.2, linestyle="--", label="衰退边界")
    ax.axvline(baseline_life, color=COLORS["orange"], linewidth=2.0, linestyle="--")
    ax.axvline(final_life, color=COLORS["red"], linewidth=2.2)
    ax.scatter([baseline_life], [0.62], s=90, color=COLORS["orange"], zorder=5)
    ax.scatter([final_life], [0.42], s=100, color=COLORS["red"], zorder=5)
    ax.text(baseline_life - 35, 0.76, f"baseline参照\n{baseline_life:.1f} d", ha="right", fontsize=10.0, color=COLORS["orange"])
    ax.text(final_life - 35, 0.30, f"final坐标\n{final_life:.1f} d", ha="right", fontsize=10.0, color=COLORS["red"])
    ax.set_xlim(0, TOTAL_LIFE)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("等效寿命坐标/天")
    ax.set_title("任务三输出对应的附件2等效寿命位置")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    ax = axes[1]
    methods = ["baseline参照", "transfer final"]
    points = [float(result["baseline_RUL_days"]), float(result["RUL_final_days"])]
    lows = [float(result["baseline_RUL_low_conservative"]), float(result["RUL_low_conservative"])]
    highs = [float(result["baseline_RUL_high_conservative"]), float(result["RUL_high_conservative"])]
    y = np.arange(2)
    for i, (point, low, high, color) in enumerate(zip(points, lows, highs, [COLORS["gray"], COLORS["blue"]])):
        ax.errorbar(point, i, xerr=[[point - low], [high - point]], fmt="o", capsize=6, linewidth=2.0, color=color)
        y_offset = 0.16 if i == 0 else -0.18
        ax.text(point + 8, i + y_offset, f"{point:.1f} d", fontsize=10, color=color)
    ax.axvline(float(result["RUL_warning_threshold_days"]), color=COLORS["yellow"], linestyle="--", linewidth=1.5, label="警戒RUL阈值")
    ax.axvline(float(result["RUL_severe_threshold_days"]), color=COLORS["red"], linestyle="--", linewidth=1.5, label="严重RUL阈值")
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_ylim(1.35, -0.35)
    ax.set_xlabel("等效剩余寿命估计/天")
    ax.set_title("RUL点估计、保守区间与规则阈值")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIG / "life_rul_panel.png", bbox_inches="tight")
    plt.close(fig)


def write_reports(result: pd.Series, truncation: pd.DataFrame, thresholds: dict[str, float]) -> None:
    baseline_mae = thresholds["task3_baseline_mae_days"]
    final_mae = thresholds["task3_final_mae_days"]
    improvement = baseline_mae - final_mae

    report = f"""# 任务四：飞轮健康管理报告

## 0 报告生成逻辑

任务四脚本 `python -m src.task4.health_management_report` 是基于任务三最终模型结果的健康预警与工程管理模块。脚本只读取任务三输出中的附件2最终预测、任务三不确定性区间和任务三截断实验误差证据，不读取其他任务的结果文件，也不重新训练寿命模型。

本版任务四采用文献中常见的 threshold / control-limit 预警思想：先判断健康阶段或健康指标是否接近阈值，再结合 RUL 及其保守区间给出预警等级和工程措施。由于 task3 当前只有 4 个附件1截断验证点和 1 个附件2当前点，样本量不足以稳定构造熵权法评价矩阵，因此本报告不采用熵权法，也不使用人工加权风险分数作为正式机制。

## 1 当前健康状态评估

评价对象为附件2飞轮。任务三附件2结果显示，当前观测时刻为第 {result['observed_day']:.0f} 天，baseline 参照健康指标约为 {result['current_HI']:.4f}。任务三文件中携带的 baseline 阶段为“{result['baseline_stage']}”，迁移残差修正后的阶段为“{result['transfer_corrected_stage']}”。

在任务三等效寿命坐标下，baseline 参照寿命位置约为 {result['baseline_equivalent_life_day']:.2f} 天；最终 RUL 对应的等效寿命位置约为 {result['final_equivalent_life_day']:.2f} 天，距离衰退边界约 {result['distance_to_decline_boundary_days']:.2f} 天。

因此，当前状态判定为：**{result['current_state']}**。

## 2 剩余寿命预测

任务三文件中携带的 baseline 参照等效剩余寿命为 {result['baseline_RUL_days']:.2f} 天；迁移残差模型给出 {result['Delta_RUL_pred_days']:.2f} 天的温和修正，得到最终等效剩余寿命：

```text
RUL_final = {result['baseline_RUL_days']:.2f} + {result['Delta_RUL_pred_days']:.2f} = {result['RUL_final_days']:.2f} 天
```

结合任务三保守区间估计，附件2当前最终等效 RUL 的保守范围约为 [{result['RUL_low_conservative']:.2f}, {result['RUL_high_conservative']:.2f}] 天，区间宽度比例约为 {result['RUL_interval_ratio']:.3f}。

在任务三截断实验中，baseline 参照的平均绝对误差约为 {baseline_mae:.2f} 天，迁移残差修正后的平均绝对误差约为 {final_mae:.2f} 天，平均降低约 {improvement:.2f} 天。这说明任务三最终模型是在保持 baseline 稳定性的基础上做系统偏差修正。

## 3 规则分级预警机制

本文不再使用手工权重风险分数。预警等级由规则触发决定，并采用“最高触发等级”为最终等级：

1. 若修正后阶段为衰退期，或 final 等效寿命越过衰退边界，则判为 3 级严重。
2. 若 final 等效寿命距衰退边界不超过 {result['boundary_warning_window_days']:.0f} 天，则判为 2 级警戒。
3. 若 RUL 保守区间下界不超过 {result['RUL_warning_threshold_days']:.0f} 天，则判为 2 级警戒；若不超过 {result['RUL_severe_threshold_days']:.0f} 天，则判为 3 级严重。
4. 若修正后阶段为退化期，或距衰退边界进入关注窗口，则至少判为 1 级关注。
5. RUL 区间宽度比例只作为不确定性触发项，不等同于失效概率。

阈值来源如下：衰退边界 2500 天来自任务三阶段体系；{result['boundary_warning_window_days']:.0f} 天警戒窗口取衰退边界处剩余寿命 1000 天的 10%，对应 control-limit 前的预警缓冲；RUL 警戒阈值 {result['RUL_warning_threshold_days']:.0f} 天对应进入衰退边界时的剩余寿命；RUL 严重阈值 {result['RUL_severe_threshold_days']:.0f} 天来自任务三截断实验中衰退阶段样本的最小真实 RUL。

附件2当前触发规则为：**{result['triggered_rule_ids']}**。触发说明为：{result['triggered_rule_descriptions']}。

综合判定附件2当前预警等级为：**{int(result['warning_level'])}级{result['warning_name']}（{result['warning_color']}预警）**。

当前不升为红色，是因为 final 等效寿命尚未越过衰退边界，且保守 RUL 下界仍约为 {result['RUL_low_conservative']:.2f} 天，高于严重阈值 {result['RUL_severe_threshold_days']:.0f} 天。

## 4 工程建议

1. 将附件2飞轮纳入加密监测状态，重点跟踪电流、温度和健康指标趋势。
2. 在后续观测中滚动更新任务三输入结果，若 final 等效寿命越过衰退边界或 RUL 区间下界快速下降，则升级为严重预警。
3. 在高载荷姿态控制任务、长时间连续姿态机动任务前，提前准备冗余飞轮或替代执行机构接管策略。
4. 若电流和温度同步出现加速上升，应优先检查润滑退化、摩擦力矩上升和轴承磨损相关风险。
5. 当前不建议立即判定为失效或停用，但建议降低不必要的高负载工况暴露。

## 5 不确定性与局限性

1. 附件2没有完整失效寿命标签，因此本文给出的 RUL 是等效工况假设下的剩余寿命预测。
2. 任务三使用轴承源域知识迁移到飞轮退化建模，但轴承与飞轮在结构、载荷和遥测变量上并不完全一致，因此迁移结果应作为趋势预警依据，而不是唯一维护依据。
3. 当前预警机制依赖任务三输出中的阶段、RUL 与区间结果；若未来获得新的在轨数据，应滚动更新模型参数和预警等级。
4. 预警等级服务于工程决策排序，不等同于失效概率的直接测量。
"""
    (OUT / "health_management_report.md").write_text(report, encoding="utf-8")

    paper_text = f"""# 任务四论文写作文本

## 健康状态预警机制

基于任务三得到的物理 baseline 约束迁移残差增强结果，本文构建面向工程决策的多级健康预警机制。该机制不重新训练寿命模型，也不直接读取前序任务的中间结果，而是以任务三输出的附件2最终 RUL、阶段判断和不确定性区间为输入，将模型结果封装为健康管理报告。

参考 PHM 文献中常见的 control-limit / threshold 预警思想，本文采用规则分级机制，而不采用人工加权风险分数。规则首先判断修正后健康阶段和 final 等效寿命是否接近衰退边界，再结合 RUL 保守区间下界与区间宽度判断是否需要升级预警。

## 附件2当前预警结果

附件2当前观测时刻为第 {result['observed_day']:.0f} 天，baseline 参照健康指标约为 {result['current_HI']:.4f}。任务三最终等效 RUL 约为 {result['RUL_final_days']:.2f} 天，保守区间约为 [{result['RUL_low_conservative']:.2f}, {result['RUL_high_conservative']:.2f}] 天；对应的 final 等效寿命坐标约为 {result['final_equivalent_life_day']:.2f} 天，距离衰退边界约 {result['distance_to_decline_boundary_days']:.2f} 天。

当前触发规则为：{result['triggered_rule_ids']}。综合判定附件2当前预警等级为 **{int(result['warning_level'])}级{result['warning_name']}（{result['warning_color']}预警）**。该等级由最高触发规则决定，不使用固定人工权重。

## 工程建议

当前附件2飞轮尚未表现为立即失效风险，但已进入退化期末段，建议进入加密监测状态。后续应重点监测电流、温度和健康指标趋势；在高负载姿态控制任务前，应准备冗余飞轮或替代执行机构接管方案。若后续 final 等效寿命越过衰退边界，或 RUL 区间下界快速下降，则应升级预警并采取更保守的任务规划。

## 局限性说明

由于附件2缺少真实失效寿命标签，本文给出的剩余寿命是等效工况假设下的模型预测结果，而非真实寿命观测值。当前 task3 样本量不足以稳定建立熵权法评价矩阵，因此本任务采用规则分级预警。预警等级是管理分级，不等同于真实失效概率。
"""
    (OUT / "paper_text.md").write_text(paper_text, encoding="utf-8")

    io_text = f"""# 任务四脚本输入输出说明

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

- 当前健康状态：{result['current_state']}
- 最终等效 RUL：{result['RUL_final_days']:.2f} 天
- 保守区间：{result['RUL_low_conservative']:.2f}-{result['RUL_high_conservative']:.2f} 天
- 当前预警等级：{int(result['warning_level'])}级{result['warning_name']}（{result['warning_color']}预警）
- 当前触发规则：{result['triggered_rule_ids']}
"""
    (OUT / "input_output_description.md").write_text(io_text, encoding="utf-8")


def write_json(result: pd.Series) -> None:
    with open(OUT / "warning_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def main() -> None:
    prediction, truncation, interval = load_task3_inputs()
    thresholds = derive_thresholds(truncation)
    rules = build_warning_rules(thresholds)
    decision = evaluate_warning(prediction, interval, thresholds)
    result_df = build_warning_result(prediction, decision)
    result = result_df.iloc[0]

    rules.to_csv(OUT / "warning_rules.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(OUT / "warning_result.csv", index=False, encoding="utf-8-sig")
    write_json(result)
    draw_warning_level(result)
    draw_risk_breakdown(result)
    draw_life_rul_panel(result)
    write_reports(result, truncation, thresholds)


if __name__ == "__main__":
    main()
