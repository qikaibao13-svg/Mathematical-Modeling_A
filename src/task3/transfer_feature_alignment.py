from __future__ import annotations

import copy
import json
import math
import os
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MM_CONTEST_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
TASK1 = PROJECT_ROOT / "outputs" / "task1"
TASK2 = PROJECT_ROOT / "outputs" / "task2"
OUT = PROJECT_ROOT / "outputs" / "task3_alignment"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

ATT1 = Path(os.environ.get("MM_CONTEST_ATT1", DATA_ROOT / "附件1 reaction_wheel_3500d_data.csv"))
ATT2 = Path(os.environ.get("MM_CONTEST_ATT2", DATA_ROOT / "附件2 reaction_wheel_1800d_data.csv"))

SEED = 20260530
SEQ_LEN = 16
SOURCE_IDS = ["Bearing3_1", "Bearing2_5", "Bearing1_1", "Bearing3_5"]
MAIN_SOURCE = "Bearing3_1"
TOTAL_LIFE_DAYS = 3500.0


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def minmax(arr: np.ndarray, lo: float, hi: float, clip: bool = True) -> np.ndarray:
    z = (np.asarray(arr, float) - lo) / max(hi - lo, 1e-12)
    return np.clip(z, 0, 1) if clip else z


def smooth(y: np.ndarray, window: int, center: bool = True) -> np.ndarray:
    return (
        pd.Series(np.asarray(y, float))
        .rolling(window, center=center, min_periods=max(2, window // 3))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(float)
    )


def add_sequence_features(df: pd.DataFrame, base_cols: list[str], group_col: str) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    feat_cols: list[str] = []
    for _, g0 in df.groupby(group_col):
        g = g0.sort_values("time_index").copy()
        for col in base_cols:
            g[f"{col}_x"] = g[col].astype(float)
            g[f"{col}_diff"] = g[col].diff().fillna(0.0)
            g[f"{col}_roll_mean"] = g[col].rolling(5, min_periods=1).mean()
            g[f"{col}_roll_std"] = g[col].rolling(5, min_periods=1).std().fillna(0.0)
            g[f"{col}_slope"] = (g[col] - g[col].shift(4)).fillna(0.0) / 4.0
            feat_cols.extend([f"{col}_x", f"{col}_diff", f"{col}_roll_mean", f"{col}_roll_std", f"{col}_slope"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True), list(dict.fromkeys(feat_cols))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            tmp[col] = tmp[col].astype(str)
    headers = list(tmp.columns)
    rows = tmp.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    def fmt(row):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows])


def load_task2_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    selected = pd.read_csv(TASK2 / "task2_selected_features.csv")
    labels = pd.read_csv(TASK2 / "task2_training_labels.csv")
    norm = pd.read_csv(TASK2 / "task2_normalized_features.csv")
    stage = pd.read_csv(TASK2 / "task2_stage_boundaries.csv")
    features = selected["feature"].tolist()
    summary = f"""# 任务三源域知识整理

## 来源

源域来自任务二最新版结果目录：`{TASK2}`。

## 核心特征集

{', '.join(features)}

## HI_bearing 构造逻辑

任务二采用 `Correlation + Monotonicity + Robustness + EWM` 筛选核心特征，并以 `J_score` 归一化权重融合构造 `HI_bearing`，随后做轻度平滑与 cumulative max 修正。

## 主源域与辅助源域

- 主源域：`Bearing3_1`
- 辅助源域：`Bearing2_5`、`Bearing1_1`、`Bearing3_5`

## BiLSTM-Attention 的角色

任务二结果汇总已将 `BiLSTM-Attention` 作为按 MAE/MSE/RMSE 口径的正式主模型。任务三复用其“BiLSTM 编码 + Attention 聚合 + RUL 回归”的源域特征提取结构，并在此基础上加入 DANN 域对抗适配。

## 主源域阶段边界

{markdown_table(stage[stage['bearing_id'].isin(SOURCE_IDS)][['bearing_id', 'tau1_life_ratio', 'tau2_life_ratio', 'method']])}
"""
    (OUT / "task3_source_domain_summary.md").write_text(summary, encoding="utf-8")
    return selected, labels, norm, stage, features


def flywheel_feature_table(path: Path, selected_features: list[str], is_attachment1: bool, ref_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path).copy()
    df["domain_unit"] = "attachment1" if is_attachment1 else "attachment2"
    df["time_index"] = np.arange(len(df))
    df["delta_I"] = df["current"] - df["current_theoretical"]
    w = 9 if is_attachment1 else 61
    for col in ["current", "current_theoretical", "temperature"]:
        df[f"{col}_s"] = smooth(df[col].to_numpy(float), w, center=True)
    if "friction_torque" in df.columns:
        df["friction_torque_s"] = smooth(df["friction_torque"].to_numpy(float), w, center=True)
    else:
        df["friction_torque_s"] = np.nan
    df["delta_I_s"] = df["current_s"] - df["current_theoretical_s"]

    if ref_stats is None:
        ref_stats = {
            "current": (float(df["current_s"].min()), float(df["current_s"].max())),
            "temperature": (float(df["temperature_s"].min()), float(df["temperature_s"].max())),
            "friction": (float(df["friction_torque_s"].min()), float(df["friction_torque_s"].max())),
            "abs_delta": (0.0, float(np.nanpercentile(np.abs(df["delta_I_s"]), 98))),
        }
    n_current = minmax(df["current_s"], *ref_stats["current"])
    n_temp = minmax(df["temperature_s"], *ref_stats["temperature"])
    if "friction_torque" in df.columns:
        n_friction = minmax(df["friction_torque_s"], *ref_stats["friction"])
    else:
        n_friction = np.clip(0.65 * n_temp + 0.35 * n_current, 0, 1)
    n_delta = minmax(np.abs(df["delta_I_s"]), *ref_stats["abs_delta"])
    roll_current_std = pd.Series(n_current).rolling(15, min_periods=2).std().fillna(0.0).to_numpy(float)
    roll_temp_std = pd.Series(n_temp).rolling(15, min_periods=2).std().fillna(0.0).to_numpy(float)
    n_var = np.clip(0.55 * n_temp + 0.25 * n_delta + 0.20 * minmax(roll_current_std, 0, max(roll_current_std.max(), 1e-6)), 0, 1)
    n_energy = np.clip(0.55 * n_current + 0.25 * n_temp + 0.20 * n_friction, 0, 1)

    mapping_values = {
        "fused_rms_vector": np.clip(0.85 * n_current + 0.15 * n_energy, 0, 1),
        "fused_rms_mean": n_current,
        "fused_std_mean": np.clip(0.70 * n_delta + 0.30 * minmax(roll_current_std, 0, max(roll_current_std.max(), 1e-6)), 0, 1),
        "fused_envelope_rms_mean": np.clip(0.65 * n_friction + 0.35 * n_temp, 0, 1),
        "fused_rms_max": np.maximum(n_current, n_energy),
        "fused_std_max": np.maximum(n_delta, minmax(roll_current_std, 0, max(roll_current_std.max(), 1e-6))),
        "fused_envelope_rms_max": np.maximum(n_friction, n_temp),
        "fused_total_energy_sum": n_energy,
        "fused_total_energy_mean": np.clip(0.65 * n_current + 0.35 * n_temp, 0, 1),
        "fused_variance_mean": n_var,
    }
    for feat in selected_features:
        df[feat] = mapping_values.get(feat, n_energy)
    df["HI_flywheel"] = np.clip(
        0.35 * mapping_values.get("fused_rms_mean", n_current)
        + 0.25 * mapping_values.get("fused_envelope_rms_mean", n_friction)
        + 0.25 * mapping_values.get("fused_total_energy_mean", n_energy)
        + 0.15 * mapping_values.get("fused_variance_mean", n_var),
        0,
        1,
    )
    df["HI_flywheel"] = np.maximum.accumulate(smooth(df["HI_flywheel"].to_numpy(float), w, center=True))
    if is_attachment1:
        df["life_ratio"] = df["day"] / df["day"].max()
        df["RUL_ratio"] = 1 - df["life_ratio"]
    else:
        df["life_ratio_observed"] = df["day"] / TOTAL_LIFE_DAYS
    return df, ref_stats


def write_feature_mapping(selected_features: list[str]) -> None:
    rows = [
        ["fused_rms_vector / fused_rms_mean", "current", "负载与能量耗散增强"],
        ["fused_std_mean", "delta_I + current局部波动", "状态波动增强"],
        ["fused_envelope_rms_mean", "friction_torque / temperature", "摩擦与局部冲击增强"],
        ["fused_total_energy_sum / mean", "current + temperature + friction", "累积耗散增强"],
        ["fused_variance_mean", "temperature + delta_I + 局部波动", "退化后不稳定性增强"],
    ]
    md = pd.DataFrame(rows, columns=["轴承侧特征", "飞轮侧对应量", "共同退化含义"])
    text = f"""# 任务三特征语义映射说明

飞轮侧不能直接使用轴承振动特征。本轮将任务二入选特征映射到同一组退化语义变量，随后对源域和目标域统一构造一阶差分、局部均值、局部波动和局部斜率。

{markdown_table(md)}

本轮任务三使用的任务二核心特征为：

{', '.join(selected_features)}
"""
    (OUT / "task3_feature_mapping_description.md").write_text(text, encoding="utf-8")


def piecewise_boundaries(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    min_seg = max(8, int(n * 0.12))
    best = None
    cands = np.unique(np.linspace(min_seg, n - min_seg - 1, min(80, n - 2 * min_seg)).astype(int))
    for i in cands:
        for j in cands:
            if j <= i + min_seg or j >= n - min_seg:
                continue
            sse = 0.0
            for a, b in [(0, i), (i, j), (j, n)]:
                coef = np.polyfit(x[a:b], y[a:b], 1)
                pred = coef[0] * x[a:b] + coef[1]
                sse += float(((y[a:b] - pred) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, i, j)
    if best is None:
        return float(x[int(n * 0.4)]), float(x[int(n * 0.75)])
    return float(x[best[1]]), float(x[best[2]])


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def transferability_analysis(labels: pd.DataFrame, stage: pd.DataFrame, att1: pd.DataFrame, att2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = np.linspace(0, 1, 201)
    b = labels[labels["bearing_id"] == MAIN_SOURCE].sort_values("life_ratio")
    bearing_hi = np.interp(grid, b["life_ratio"], b["HI_bearing"])
    fly_hi = np.interp(grid, att1["life_ratio"], att1["HI_flywheel"])
    att2_grid_x = np.clip(att2["day"].to_numpy(float) / TOTAL_LIFE_DAYS, 0, 1)
    att2_hi = np.interp(grid, att2_grid_x, att2["HI_flywheel"], left=att2["HI_flywheel"].iloc[0], right=att2["HI_flywheel"].iloc[-1])
    curve = pd.DataFrame({"life_ratio": grid, "Bearing3_1_HI": bearing_hi, "flywheel_attachment1_HI": fly_hi, "flywheel_attachment2_observed_HI": att2_hi})
    curve.to_csv(OUT / "task3_HI_curve_comparison.csv", index=False)
    bt = stage[stage["bearing_id"] == MAIN_SOURCE].iloc[0]
    f_tau1, f_tau2 = piecewise_boundaries(att1["life_ratio"].to_numpy(float), att1["HI_flywheel"].to_numpy(float))
    stage_cmp = pd.DataFrame(
        [
            {
                "source": "Bearing3_1",
                "tau1_life_ratio": float(bt["tau1_life_ratio"]),
                "tau2_life_ratio": float(bt["tau2_life_ratio"]),
            },
            {"source": "flywheel_attachment1", "tau1_life_ratio": f_tau1, "tau2_life_ratio": f_tau2},
        ]
    )
    stage_cmp["tau1_diff_vs_bearing"] = stage_cmp["tau1_life_ratio"] - float(bt["tau1_life_ratio"])
    stage_cmp["tau2_diff_vs_bearing"] = stage_cmp["tau2_life_ratio"] - float(bt["tau2_life_ratio"])
    stage_cmp.to_csv(OUT / "task3_stage_boundary_comparison.csv", index=False)
    sim = pd.DataFrame(
        [
            {
                "pair": "Bearing3_1_HI_vs_flywheel_attachment1_HI",
                "pearson": float(pearsonr(bearing_hi, fly_hi).statistic),
                "spearman": float(spearmanr(bearing_hi, fly_hi).statistic),
                "dtw_distance": dtw_distance(bearing_hi, fly_hi),
                "euclidean_distance": float(np.linalg.norm(bearing_hi - fly_hi) / math.sqrt(len(grid))),
                "tau1_abs_diff": abs(float(bt["tau1_life_ratio"]) - f_tau1),
                "tau2_abs_diff": abs(float(bt["tau2_life_ratio"]) - f_tau2),
            },
            {
                "pair": "flywheel_attachment1_HI_vs_attachment2_observed_HI",
                "pearson": float(pearsonr(fly_hi, att2_hi).statistic),
                "spearman": float(spearmanr(fly_hi, att2_hi).statistic),
                "dtw_distance": dtw_distance(fly_hi, att2_hi),
                "euclidean_distance": float(np.linalg.norm(fly_hi - att2_hi) / math.sqrt(len(grid))),
                "tau1_abs_diff": np.nan,
                "tau2_abs_diff": np.nan,
            },
        ]
    )
    sim.to_csv(OUT / "task3_similarity_metrics.csv", index=False)
    text = f"""# 任务三小任务1：可迁移性分析

## 物理一致性

轴承侧退化主要表现为润滑退化和接触磨损增强，导致振动 RMS、标准差、包络 RMS 与能量类特征上升。飞轮侧退化同样与润滑劣化、摩擦力矩增大有关，表现为电流、温度、摩擦负载与电流偏差增强。因此二者原始观测信号不同，但共同对应“润滑劣化-摩擦增强-能量耗散增加”的退化语义链。

## HI 与模板形态一致性

源域采用 `Bearing3_1` 的 `HI_bearing`，目标域采用附件1构造的 `HI_flywheel`。统一到 `[0,1]` 生命周期坐标后，二者的定量相似性如下：

{markdown_table(sim)}

## 阶段边界比较

{markdown_table(stage_cmp)}

## 判断

Pearson/Spearman 反映整体趋势一致性，DTW/欧氏距离反映模板形态差异。阶段边界存在明显时间尺度差异，说明不能直接复制轴承寿命比例到飞轮；但趋势单调性和退化语义一致，支持使用 DANN/ADDA 对齐退化表示，而不是简单套用原始时间模板。
"""
    (OUT / "task3_transferability_analysis.md").write_text(text, encoding="utf-8")
    return curve, stage_cmp, sim


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], unit_col: str, label_col: str | None = "RUL_ratio", seq_len: int = SEQ_LEN):
        self.x, self.y = [], []
        for _, g0 in df.groupby(unit_col):
            g = g0.sort_values("time_index").reset_index(drop=True)
            arr = g[feature_cols].to_numpy(np.float32)
            labels = g[label_col].to_numpy(np.float32) if label_col and label_col in g.columns else None
            for i in range(len(g)):
                start = max(0, i - seq_len + 1)
                seq = arr[start : i + 1]
                if len(seq) < seq_len:
                    seq = np.vstack([np.repeat(seq[:1], seq_len - len(seq), axis=0), seq])
                self.x.append(seq)
                self.y.append(float(labels[i]) if labels is not None else np.nan)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class EncoderAttention(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        e = self.attn(h).squeeze(-1)
        a = torch.softmax(e, dim=1)
        z = (h * a.unsqueeze(-1)).sum(dim=1)
        return z, a


class DANNRUL(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48):
        super().__init__()
        self.encoder = EncoderAttention(n_features, hidden)
        self.regressor = nn.Sequential(nn.Linear(hidden * 2, 48), nn.ReLU(), nn.Dropout(0.1), nn.Linear(48, 1), nn.Sigmoid())
        self.domain = nn.Sequential(nn.Linear(hidden * 2, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, x, alpha: float = 0.0):
        z, a = self.encoder(x)
        y = self.regressor(z).squeeze(-1)
        d = self.domain(GradientReverse.apply(z, alpha)).squeeze(-1)
        return y, d, z, a


def build_model_inputs(norm: pd.DataFrame, att1: pd.DataFrame, att2: pd.DataFrame, selected_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    src = norm[norm["bearing_id"].isin(SOURCE_IDS)].copy()
    src["domain_unit"] = src["bearing_id"]
    src["time_index"] = src["sample_index"].astype(int) - 1
    src, feat_cols = add_sequence_features(src, selected_features, "domain_unit")
    t1, tfeat = add_sequence_features(att1.copy(), selected_features, "domain_unit")
    t2, _ = add_sequence_features(att2.copy(), selected_features, "domain_unit")
    assert feat_cols == tfeat
    scaler = StandardScaler()
    scaler.fit(pd.concat([src[feat_cols], t1[feat_cols]], ignore_index=True))
    for df in [src, t1, t2]:
        df[feat_cols] = scaler.transform(df[feat_cols])
    return src, t1, t2, feat_cols


def mmd_linear(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def collect_embeddings(model: DANNRUL, df: pd.DataFrame, feat_cols: list[str], unit_col: str, max_batches: int = 60) -> np.ndarray:
    ds = SeqDataset(df, feat_cols, unit_col, "RUL_ratio" if "RUL_ratio" in df.columns else None)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for k, (xb, _) in enumerate(loader):
            _, _, z, _ = model(xb, alpha=0.0)
            chunks.append(z.numpy())
            if k >= max_batches:
                break
    return np.vstack(chunks)


def train_dann(src: pd.DataFrame, t1: pd.DataFrame, t2: pd.DataFrame, feat_cols: list[str]) -> tuple[DANNRUL, pd.DataFrame, pd.DataFrame]:
    seed_everything()
    model = DANNRUL(len(feat_cols), hidden=48)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    src_ds = SeqDataset(src, feat_cols, "domain_unit", "RUL_ratio")
    t1_ds = SeqDataset(t1, feat_cols, "domain_unit", "RUL_ratio")
    # DANN adaptation uses attachment 1 as labelled target-domain life-cycle data.
    # Attachment 2 is held out for application to avoid pulling the domain discriminator
    # toward an incomplete in-orbit trajectory.
    tunlab = t1.copy()
    tunlab_ds = SeqDataset(tunlab, feat_cols, "domain_unit", None)

    src_loader = DataLoader(src_ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    t1_loader = DataLoader(t1_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED + 1))
    tu_loader = DataLoader(tunlab_ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED + 2))

    logs = []
    # Stage 1: source supervised training.
    for epoch in range(14):
        model.train()
        total, count = 0.0, 0
        for xb, yb in src_loader:
            opt.zero_grad()
            pred, _, _, _ = model(xb, alpha=0.0)
            beta = 1.0 + 2.0 * (1.0 - yb)
            loss = (beta * (pred - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        logs.append({"stage": "source_supervised", "epoch": epoch + 1, "loss_rul": total / count, "loss_domain": np.nan, "domain_acc": np.nan, "lambda": 0.0})

    pre_model = copy.deepcopy(model)
    before_src = collect_embeddings(pre_model, src, feat_cols, "domain_unit")
    before_t = collect_embeddings(pre_model, t1, feat_cols, "domain_unit")

    # Stage 2: DANN adaptation. Target attachment1 has RUL labels; attachment2 is domain-only.
    steps = max(len(src_loader), len(t1_loader), len(tu_loader))
    for epoch in range(28):
        model.train()
        src_iter, t1_iter, tu_iter = cycle(src_loader), cycle(t1_loader), cycle(tu_loader)
        loss_rul_sum, loss_dom_sum, acc_sum, count = 0.0, 0.0, 0.0, 0
        p = epoch / max(1, 27)
        lam = 2 / (1 + math.exp(-8 * p)) - 1
        for _ in range(steps):
            xs, ys = next(src_iter)
            xt, yt = next(t1_iter)
            xu, _ = next(tu_iter)
            opt.zero_grad()
            ps, ds, zs, _ = model(xs, alpha=lam)
            pt, dt_label, zt, _ = model(xt, alpha=lam)
            _, du, _, _ = model(xu, alpha=lam)
            beta_s = 1.0 + 2.0 * (1.0 - ys)
            beta_t = 1.0 + 2.0 * (1.0 - yt)
            loss_rul = (beta_s * (ps - ys) ** 2).mean() + 0.70 * (beta_t * (pt - yt) ** 2).mean()
            logits = torch.cat([ds, dt_label, du])
            labels = torch.cat([torch.zeros_like(ds), torch.ones_like(dt_label), torch.ones_like(du)])
            loss_dom = bce(logits, labels)
            loss_moment = ((zs.mean(dim=0) - zt.mean(dim=0)) ** 2).mean()
            loss = loss_rul + 0.35 * loss_dom + 1.00 * loss_moment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pred_dom = (torch.sigmoid(logits) >= 0.5).float()
            acc = (pred_dom == labels).float().mean().item()
            n = len(logits)
            loss_rul_sum += float(loss_rul.item()) * n
            loss_dom_sum += float(loss_dom.item()) * n
            acc_sum += acc * n
            count += n
        logs.append({"stage": "dann_adaptation", "epoch": epoch + 1, "loss_rul": loss_rul_sum / count, "loss_domain": loss_dom_sum / count, "domain_acc": acc_sum / count, "lambda": lam})

    after_src = collect_embeddings(model, src, feat_cols, "domain_unit")
    after_t1 = collect_embeddings(model, t1, feat_cols, "domain_unit")
    after_t2 = collect_embeddings(model, t2, feat_cols, "domain_unit")
    align = pd.DataFrame(
        [
            {"metric": "centroid_distance_source_attachment1_before", "value": mmd_linear(before_src, before_t)},
            {"metric": "centroid_distance_source_attachment1_after", "value": mmd_linear(after_src, after_t1)},
            {"metric": "centroid_distance_source_attachment2_after", "value": mmd_linear(after_src, after_t2)},
            {"metric": "embedding_dim", "value": after_src.shape[1]},
        ]
    )
    log_df = pd.DataFrame(logs)
    log_df.to_csv(OUT / "task3_dann_training_log.csv", index=False)
    align.to_csv(OUT / "task3_domain_alignment_metrics.csv", index=False)
    return model, log_df, align


def predict_dataframe(model: DANNRUL, df: pd.DataFrame, feat_cols: list[str], unit_col: str, out_name: str) -> pd.DataFrame:
    rows = []
    model.eval()
    for unit, g0 in df.groupby(unit_col):
        g = g0.sort_values("time_index").reset_index(drop=True)
        arr = g[feat_cols].to_numpy(np.float32)
        with torch.no_grad():
            for i in range(len(g)):
                start = max(0, i - SEQ_LEN + 1)
                seq = arr[start : i + 1]
                if len(seq) < SEQ_LEN:
                    seq = np.vstack([np.repeat(seq[:1], SEQ_LEN - len(seq), axis=0), seq])
                pred, _, _, att = model(torch.tensor(seq[None, :, :], dtype=torch.float32), alpha=0.0)
                item = g.iloc[i].to_dict()
                item["pred_RUL_ratio_raw"] = float(pred.item())
                item["attention_last_weight"] = float(att.numpy()[0, -1])
                rows.append(item)
    out = pd.DataFrame(rows)
    out["pred_RUL_ratio"] = out.groupby(unit_col)["pred_RUL_ratio_raw"].transform(lambda s: np.minimum.accumulate(s.to_numpy(float)))
    out["pred_RUL_days"] = out["pred_RUL_ratio"] * TOTAL_LIFE_DAYS
    out.to_csv(OUT / out_name, index=False)
    return out


def shifted_exp(t: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    return c + a * (np.exp(b * t) - 1)


def fit_exp_given_b(t: np.ndarray, y: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    x = np.exp(b * t) - 1
    design = np.column_stack([np.ones_like(t), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ coef
    return np.array([coef[0], coef[1], b]), float(np.sum((y - yhat) ** 2))


def fit_exp(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    grid = np.linspace(np.log(1e-7), np.log(1e-2), 260)
    best = None
    for lb in grid:
        p, sse = fit_exp_given_b(t, y, float(np.exp(lb)))
        if p[1] > 0 and (best is None or sse < best[0]):
            best = (sse, p)
    if best is None:
        return np.array([y[0], max(y[-1] - y[0], 1e-6), 1e-4])
    return best[1]


def inverse_exp(y: float, popt: np.ndarray) -> float:
    c, a, b = popt
    inside = (y - c) / max(a, 1e-12) + 1
    return float(np.log(max(inside, 1e-12)) / b)


def comparison_experiments(att1_pred: pd.DataFrame, att2_pred: pd.DataFrame, att1_proc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fail_value = float(att1_proc["current_theoretical_s"].iloc[-1])
    rows = []
    for frac in [0.6, 0.7, 0.8, 0.9]:
        idx = int(np.argmin(np.abs(att1_proc["life_ratio"] - frac)))
        sub = att1_proc.iloc[: idx + 1]
        popt = fit_exp(sub["day"].to_numpy(float), sub["current_theoretical_s"].to_numpy(float))
        pred_fail = inverse_exp(fail_value, popt)
        baseline_rul = max(pred_fail - float(sub["day"].iloc[-1]), 0)
        pred_row = att1_pred.iloc[idx]
        transfer_rul = float(pred_row["pred_RUL_days"])
        true_rul = TOTAL_LIFE_DAYS - float(sub["day"].iloc[-1])
        rows.append(
            {
                "fraction": frac,
                "observed_day": float(sub["day"].iloc[-1]),
                "true_RUL_days": true_rul,
                "baseline_RUL_days": baseline_rul,
                "baseline_abs_error_days": abs(baseline_rul - true_rul),
                "transfer_RUL_days": transfer_rul,
                "transfer_abs_error_days": abs(transfer_rul - true_rul),
            }
        )
    trunc = pd.DataFrame(rows)
    trunc.to_csv(OUT / "task3_attachment1_truncation_experiment.csv", index=False)
    with open(TASK1 / "current_state_and_rul.json", "r", encoding="utf-8") as f:
        baseline_state = json.load(f)
    last = att2_pred.sort_values("day").iloc[-1]
    transfer_rul = float(last["pred_RUL_days"])
    eq_life = TOTAL_LIFE_DAYS - transfer_rul
    tau1, tau2 = baseline_state["tau1"], baseline_state["tau2"]
    stage = "健康期" if eq_life < tau1 else ("退化期" if eq_life < tau2 else "衰退期")
    final = pd.DataFrame(
        [
            {
                "object": "attachment2_current",
                "baseline_stage": baseline_state["stage_conclusion"],
                "baseline_RUL_days": baseline_state["rul_point"],
                "transfer_stage": stage,
                "transfer_RUL_days": transfer_rul,
                "transfer_equivalent_life_day": eq_life,
                "baseline_HI2": baseline_state["current_HI2"],
                "transfer_HI": float(np.clip(0.5 * last["HI_flywheel"] + 0.5 * (1 - last["pred_RUL_ratio"]), 0, 1)),
            }
        ]
    )
    final.to_csv(OUT / "task3_baseline_vs_transfer_comparison.csv", index=False)
    assess = f"""# 附件2迁移模型状态评估

附件2末端 day={last['day']:.0f}。

- 任务一 baseline：阶段 `{baseline_state['stage_conclusion']}`，RUL `{baseline_state['rul_point']:.2f}` 天。
- 任务三 transfer-enhanced：阶段 `{stage}`，RUL `{transfer_rul:.2f}` 天，等效寿命位置 `{eq_life:.2f}` 天。
- transfer HI：{final.loc[0, 'transfer_HI']:.4f}

解释：迁移模型使用任务二 BiLSTM-Attention 编码结构和 DANN 域对齐后的飞轮语义特征，不直接使用附件2未来标签。阶段判断仍以任务一附件1变化点作为飞轮本体阶段坐标。
"""
    (OUT / "task3_attachment2_stage_assessment.md").write_text(assess, encoding="utf-8")
    summary = f"""# baseline vs transfer 最终比较

## 附件1截断实验

{markdown_table(trunc)}

## 附件2最终状态

{markdown_table(final)}

## 结论

在附件1截断点上，迁移模型和任务一 baseline 使用同一真实 RUL 口径比较。当前结果表明，任务一 baseline 作为飞轮本体模型在多数截断点上误差更小；迁移模型在 80% 截断点更优，并在附件2上给出更保守的 RUL 判断。因而本轮迁移学习的作用更适合作为跨域风险修正和辅助判据，而不是直接替代任务一 baseline。
"""
    (OUT / "alignment_summary.md").write_text(summary, encoding="utf-8")
    return trunc, final, pd.DataFrame([baseline_state])


def transfer_architecture_report(log_df: pd.DataFrame, align: pd.DataFrame) -> None:
    text = f"""# 任务三迁移模型结构与训练说明

## 参考论文复用

- 主参考 1 `Unsupervised Domain Adaptation based Remaining Useful Life Prediction of Rolling Element Bearings`：复用 Bi-LSTM 时序特征提取、DANN 域对抗适配、源/目标表示对齐思想。
- 主参考 2 `Cross-Domain Remaining Useful Life Prediction Based on Adversarial Training`：借鉴“源域监督训练 → 目标域适配 → 目标域预测”的三阶段流程。

## 本题适配

轴承和飞轮原始传感量不同，因此先进行特征语义映射，将轴承 RMS/std/envelope/energy 类特征映射为飞轮 current、delta_I、temperature、friction_torque 及其局部统计。随后使用同一 BiLSTM-Attention 编码器结构进行表示学习。

## 模型结构

```text
semantic feature sequence
 -> BiLSTM encoder
 -> Attention pooling
 -> RUL regressor
 -> Gradient Reversal
 -> Domain discriminator
```

## 损失函数

```text
L = L_RUL(source) + 0.70 * L_RUL(attachment1) + 0.35 * L_domain + L_moment
```

其中 `L_domain` 通过 Gradient Reversal 使源域轴承表示与附件1飞轮全寿命表示对齐；`L_moment` 约束源/目标 batch 表示均值接近，用于稳定小样本目标域下的对抗训练。附件2不使用 RUL 标签，也不参与训练，仅作为最终在轨应用对象。

## 训练日志摘要

{markdown_table(log_df.tail(8))}

## 域对齐指标

{markdown_table(align)}
"""
    (OUT / "task3_transfer_model_architecture.md").write_text(text, encoding="utf-8")


def make_figures(curve: pd.DataFrame, stage_cmp: pd.DataFrame, log_df: pd.DataFrame, trunc: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve["life_ratio"], curve["Bearing3_1_HI"], label="Bearing3_1 HI")
    plt.plot(curve["life_ratio"], curve["flywheel_attachment1_HI"], label="Flywheel attachment1 HI")
    plt.plot(curve["life_ratio"], curve["flywheel_attachment2_observed_HI"], label="Flywheel attachment2 observed HI", alpha=0.8)
    plt.xlabel("life_ratio")
    plt.ylabel("normalized HI")
    plt.title("Source and Target HI Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task3_hi_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(stage_cmp))
    plt.bar(x - 0.18, stage_cmp["tau1_life_ratio"], width=0.35, label="tau1")
    plt.bar(x + 0.18, stage_cmp["tau2_life_ratio"], width=0.35, label="tau2")
    plt.xticks(x, stage_cmp["source"], rotation=15)
    plt.ylabel("life_ratio")
    plt.title("Stage Boundary Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task3_stage_boundary_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    for stage, g in log_df.groupby("stage"):
        plt.plot(g["epoch"], g["loss_rul"], label=f"{stage} RUL")
        if g["loss_domain"].notna().any():
            plt.plot(g["epoch"], g["loss_domain"], label=f"{stage} domain")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("DANN Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task3_dann_training_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(trunc["fraction"], trunc["baseline_abs_error_days"], marker="o", label="baseline")
    plt.plot(trunc["fraction"], trunc["transfer_abs_error_days"], marker="o", label="transfer")
    plt.xlabel("attachment1 truncation fraction")
    plt.ylabel("absolute RUL error (days)")
    plt.title("Baseline vs Transfer Truncation Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task3_baseline_vs_transfer_error.png", dpi=220)
    plt.close()


def final_report(selected: list[str], sim: pd.DataFrame, comparison: pd.DataFrame, final: pd.DataFrame, trunc: pd.DataFrame, align: pd.DataFrame) -> None:
    text = f"""# 任务三完整执行报告：跨领域迁移学习

## 方法主线

任务三以任务二 `BiLSTM-Attention` 为源域特征提取器基础，复用 DANN/ADDA 迁移学习文献中的“Bi-LSTM 时序编码 + 域对抗对齐 + 目标域 RUL 预测”思想。训练流程采用三阶段：源域监督训练、目标域域适配、目标域预测。

## 源域与目标域

- 源域：任务二轴承样本，主源域 `Bearing3_1`，辅助源域 `Bearing2_5/Bearing1_1/Bearing3_5`。
- 目标域：飞轮附件1和附件2。
- 核心语义特征：{', '.join(selected)}

## 可迁移性分析摘要

{markdown_table(sim)}

## 域对齐结果

{markdown_table(align)}

## 附件1截断实验

{markdown_table(trunc)}

## 附件2最终预测

{markdown_table(final)}

## 结论

轴承与飞轮在原始观测层不同，但在“润滑劣化-摩擦增强-能量耗散增加”退化语义上可对齐。DANN 迁移模型利用任务二 BiLSTM-Attention 的时序编码能力，并通过域判别器约束源域和目标域表示接近。最终输出可与任务一 baseline 比较的飞轮 RUL 与阶段判断。
"""
    (OUT / "task3_execution_report.md").write_text(text, encoding="utf-8")
    paper = f"""# 可写入论文的任务三方法与结果

为将轴承源域退化知识迁移到飞轮目标域，本文首先分析二者的物理一致性：轴承侧润滑退化和接触磨损会导致振动能量、包络能量和冲击特征增强；飞轮侧润滑退化和摩擦力矩增大会导致电流、温度和摩擦负载上升。虽然原始信号形式不同，但二者均可归纳为“润滑劣化—摩擦增强—能量耗散增加”的渐进退化过程。

在特征层面，本文将任务二筛选出的 RMS、标准差、包络 RMS 和能量类轴承特征映射到飞轮的电流、电流偏差、温度和摩擦力矩等语义变量，并构造一阶差分、局部均值、局部波动和局部斜率特征。在此基础上，以任务二 `BiLSTM-Attention` 的时序编码结构作为源域特征提取器基础，构建 DANN 域对抗迁移模型。模型先在轴承源域上进行监督 RUL 训练，再引入飞轮附件1进行目标域适配，其中附件1提供目标域 RUL 监督，附件2不参与训练，仅作为最终在轨应用对象。

实验中，本文将迁移模型与任务一飞轮 baseline 进行对比。附件1截断实验表明，任务一 baseline 在多数截断点上更准确，而迁移模型在 80% 截断点取得更小误差，并在附件2末端给出更保守的 RUL 与阶段判断。因此，当前迁移结果更适合作为飞轮本体 baseline 的跨域辅助修正，而不是完全替代 baseline。
"""
    (OUT / "task3_paper_text.md").write_text(paper, encoding="utf-8")


def main() -> None:
    seed_everything()
    selected_df, labels, norm, stage, selected = load_task2_source()
    write_feature_mapping(selected)
    att1, ref_stats = flywheel_feature_table(ATT1, selected, is_attachment1=True)
    att2, _ = flywheel_feature_table(ATT2, selected, is_attachment1=False, ref_stats=ref_stats)
    att1.to_csv(OUT / "task3_flywheel_feature_table_attachment1.csv", index=False)
    att2.to_csv(OUT / "task3_flywheel_feature_table_attachment2.csv", index=False)
    curve, stage_cmp, sim = transferability_analysis(labels, stage, att1, att2)
    # Attachment 2 is sampled daily, while attachment 1 is sampled every 20 days.
    # For the sequence model, use a 20-day cadence to keep the temporal window comparable.
    att2_model = att2[(att2["day"] % 20 == 0) | (att2["day"] == att2["day"].max())].copy()
    att2_model["time_index"] = np.arange(len(att2_model))
    src, t1, t2, feat_cols = build_model_inputs(norm, att1, att2_model, selected)
    model, log_df, align = train_dann(src, t1, t2, feat_cols)
    transfer_architecture_report(log_df, align)
    att1_pred = predict_dataframe(model, t1, feat_cols, "domain_unit", "task3_attachment1_transfer_predictions.csv")
    att2_pred = predict_dataframe(model, t2, feat_cols, "domain_unit", "task3_attachment2_predictions.csv")
    trunc, final, baseline_state = comparison_experiments(att1_pred, att2_pred, att1)
    make_figures(curve, stage_cmp, log_df, trunc)
    final_report(selected, sim, pd.read_csv(TASK2 / "task2_main_model_comparison.csv"), final, trunc, align)
    print("TASK3_DONE", OUT)
    print("attachment2_transfer", final.to_string(index=False))
    print("truncation", trunc.to_string(index=False))


if __name__ == "__main__":
    main()
