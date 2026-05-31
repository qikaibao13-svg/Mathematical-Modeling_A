from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MM_CONTEST_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
ATT1 = Path(os.environ.get("MM_CONTEST_ATT1", DATA_ROOT / "附件1 reaction_wheel_3500d_data.csv"))
ATT2 = Path(os.environ.get("MM_CONTEST_ATT2", DATA_ROOT / "附件2 reaction_wheel_1800d_data.csv"))
OUT = PROJECT_ROOT / "outputs" / "task1"
OUT.mkdir(parents=True, exist_ok=True)


def shifted_exp(t: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    return c + a * (np.exp(b * t) - 1.0)


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "R2": 1 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "MAE": float(np.mean(np.abs(resid))),
    }


def minmax(x: pd.Series | np.ndarray, lo: float, hi: float, clip: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    z = (arr - lo) / (hi - lo)
    return np.clip(z, 0, 1) if clip else z


def smooth_series(y: pd.Series, window: int, poly: int = 2) -> np.ndarray:
    del poly
    if window % 2 == 0:
        window += 1
    return (
        y.astype(float)
        .rolling(window=window, center=True, min_periods=max(2, window // 3))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(float)
    )


def fit_exp_given_b(t: np.ndarray, y: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.exp(b * t) - 1.0
    design = np.column_stack([np.ones_like(t), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    c, a = coef
    yhat = c + a * x
    sse = float(np.sum((y - yhat) ** 2))
    return np.array([c, a, b], dtype=float), yhat, sse


def fit_exp(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, None, np.ndarray, dict[str, float]]:
    # One-dimensional search in log(b). For fixed b, c and a are linear least-squares parameters.
    def objective(log_b: float) -> float:
        b = float(np.exp(log_b))
        params, _, sse = fit_exp_given_b(t, y, b)
        if params[1] <= 0:
            return float("inf")
        return sse

    grid = np.linspace(np.log(1e-7), np.log(1e-2), 500)
    vals = np.array([objective(z) for z in grid])
    k = int(np.nanargmin(vals))
    lo = grid[max(0, k - 2)]
    hi = grid[min(len(grid) - 1, k + 2)]
    gr = (np.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(120):
        if f1 > f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = objective(x2)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = objective(x1)
    b = float(np.exp((lo + hi) / 2))
    popt, yhat, _ = fit_exp_given_b(t, y, b)
    yhat = shifted_exp(t, *popt)
    return popt, None, yhat, metrics(y, yhat)


def fit_continuous_piecewise(
    t: np.ndarray,
    y: np.ndarray,
    min_len: int = 20,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    n = len(t)
    for i in range(min_len, n - 2 * min_len + 1):
        for j in range(i + min_len, n - min_len + 1):
            tau1, tau2 = t[i], t[j]
            x = np.column_stack(
                [
                    np.ones_like(t),
                    t,
                    np.maximum(0, t - tau1),
                    np.maximum(0, t - tau2),
                ]
            )
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            yhat = x @ coef
            slopes = np.array([coef[1], coef[1] + coef[2], coef[1] + coef[2] + coef[3]])
            # Enforce the engineering interpretation: degradation rate should not decrease.
            if not (slopes[0] <= slopes[1] <= slopes[2]):
                continue
            sse = float(np.sum((y - yhat) ** 2))
            if best is None or sse < best["sse"]:
                best = {
                    "tau1": float(tau1),
                    "tau2": float(tau2),
                    "coef": coef,
                    "yhat": yhat,
                    "slopes": slopes,
                    "sse": sse,
                    "metrics": metrics(y, yhat),
                    "break_indices": [int(i), int(j)],
                }
    if best is None:
        raise RuntimeError("No valid piecewise fit found.")
    return best


def piecewise_predict(t: np.ndarray, tau1: float, tau2: float, coef: np.ndarray) -> np.ndarray:
    x = np.column_stack(
        [np.ones_like(t), t, np.maximum(0, t - tau1), np.maximum(0, t - tau2)]
    )
    return x @ coef


def inverse_exp(y: float, popt: np.ndarray) -> float:
    c, a, b = popt
    inside = (y - c) / a + 1.0
    if inside <= 0:
        return float("nan")
    return float(np.log(inside) / b)


def pseudo_online_predictions(df: pd.DataFrame, y_col: str, fail_value: float) -> list[dict[str, float]]:
    rows = []
    for frac in [0.6, 0.7, 0.8]:
        k = int(round(len(df) * frac))
        sub = df.iloc[:k]
        popt, _, _, m = fit_exp(sub["day"].to_numpy(float), sub[y_col].to_numpy(float))
        t_fail = inverse_exp(fail_value, popt)
        rows.append(
            {
                "used_fraction": frac,
                "last_observed_day": float(sub["day"].iloc[-1]),
                "predicted_failure_day": t_fail,
                "failure_day_error": t_fail - 3500,
                **m,
            }
        )
    return rows


def exp_bootstrap_inverse_ci(
    t: np.ndarray,
    y: np.ndarray,
    y_obs: float,
    base_fit: dict[str, object],
    n: int = 1500,
) -> np.ndarray:
    rng = np.random.default_rng(20260528)
    yhat = np.asarray(base_fit["yhat"], dtype=float)
    resid = y - yhat
    rmse = float(base_fit["metrics"]["RMSE"])
    samples = []
    for _ in range(n):
        y_star = yhat + rng.choice(resid, size=len(resid), replace=True)
        try:
            popt, _, _, _ = fit_exp(t, y_star)
        except Exception:
            continue
        y_draw = rng.normal(y_obs, rmse)
        ti = inverse_exp(float(y_draw), popt)
        if np.isfinite(ti):
            samples.append(ti)
    if len(samples) < 100:
        base_t = inverse_exp(y_obs, np.asarray(base_fit["params"], dtype=float))
        return np.array([base_t - 20, base_t, base_t + 20], dtype=float)
    return np.quantile(np.asarray(samples), [0.025, 0.5, 0.975])


def interpolate_daily(template: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    days = np.arange(int(template["day"].min()), int(template["day"].max()) + 1)
    out = pd.DataFrame({"day": days})
    for col in cols:
        out[col] = np.interp(days, template["day"], template[col])
    return out


def sliding_template_match(
    template_daily: pd.DataFrame,
    target: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    weights: np.ndarray,
) -> dict[str, object]:
    target_seg = target.iloc[-window:][feature_cols].to_numpy(float)
    best = None
    scores = []
    for end_day in range(window - 1, int(template_daily["day"].max()) + 1):
        seg = template_daily.iloc[end_day - window + 1 : end_day + 1][feature_cols].to_numpy(float)
        if seg.shape != target_seg.shape:
            continue
        diff = (seg - target_seg) * weights
        score = float(np.sqrt(np.mean(diff**2)))
        scores.append((end_day, score))
        if best is None or score < best[1]:
            best = (end_day, score)
    scores_df = pd.DataFrame(scores, columns=["end_day", "score"])
    min_score = scores_df["score"].min()
    # A tolerance floor avoids a falsely zero-width interval on synthetic, near-identical tracks.
    tolerance = max(min_score * 1.10, 0.004)
    candidates = scores_df[scores_df["score"] <= tolerance]
    return {
        "window": window,
        "best_end_day": float(best[0]),
        "best_score": float(best[1]),
        "candidate_low": float(candidates["end_day"].min()),
        "candidate_high": float(candidates["end_day"].max()),
        "candidate_count": int(len(candidates)),
        "scores": scores_df,
    }


def _scale(vals: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
    if hi == lo:
        return np.full_like(vals, (out_lo + out_hi) / 2, dtype=float)
    return out_lo + (vals - lo) / (hi - lo) * (out_hi - out_lo)


def write_svg_plot(
    path: Path,
    title: str,
    series: list[dict[str, object]],
    verticals: list[dict[str, object]] | None = None,
    width: int = 980,
    height: int = 520,
    y_label: str = "",
) -> None:
    verticals = verticals or []
    left, right, top, bottom = 70, 25, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    ys = np.concatenate([np.asarray(s["y"], dtype=float) for s in series])
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = (y_max - y_min) * 0.08 or 1
    y_min -= pad
    y_max += pad

    def sx(x: np.ndarray | float) -> np.ndarray:
        return _scale(np.asarray(x, dtype=float), x_min, x_max, left, left + plot_w)

    def sy(y: np.ndarray | float) -> np.ndarray:
        return _scale(np.asarray(y, dtype=float), y_min, y_max, top + plot_h, top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for frac in np.linspace(0, 1, 6):
        x = left + frac * plot_w
        val = x_min + frac * (x_max - x_min)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 38}" font-family="Arial" font-size="11" text-anchor="middle">{val:.0f}</text>')
    for frac in np.linspace(0, 1, 5):
        y = top + plot_h - frac * plot_h
        val = y_min + frac * (y_max - y_min)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{val:.3g}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e8e8e8"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 10}" font-family="Arial" font-size="13" text-anchor="middle">Day</text>')
    if y_label:
        parts.append(f'<text x="18" y="{top + plot_h / 2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})">{y_label}</text>')

    for v in verticals:
        x = float(sx(float(v["x"])))
        color = v.get("color", "#888")
        label = v.get("label", "")
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{color}" stroke-dasharray="5,5"/>')
        if label:
            parts.append(f'<text x="{x + 5:.1f}" y="{top + 16}" font-family="Arial" font-size="12" fill="{color}">{label}</text>')

    legend_x, legend_y = left + 8, top + 18
    for i, s in enumerate(series):
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        pts = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in zip(sx(x), sy(y)))
        color = str(s.get("color", "#1f77b4"))
        dash = ' stroke-dasharray="6,4"' if s.get("dash") else ""
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{pts}"/>')
        ly = legend_y + i * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{legend_x + 30}" y="{ly + 4}" font-family="Arial" font-size="12">{s["label"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    df1 = pd.read_csv(ATT1)
    df2 = pd.read_csv(ATT2)

    for df in [df1, df2]:
        df["I_obs"] = df["current"]
        df["I_theo"] = df["current_theoretical"]
        df["T"] = df["temperature"]
        df["delta_I"] = df["current"] - df["current_theoretical"]
    df1["F"] = df1["friction_torque"]

    # Centered moving-average smoothing: roughly 180 days for attachment 1 and 61 days for attachment 2.
    for col in ["current", "current_theoretical", "temperature"]:
        df1[f"{col}_s"] = smooth_series(df1[col], window=9, poly=2)
        df2[f"{col}_s"] = smooth_series(df2[col], window=61, poly=2)

    ranges = {
        "current": (float(df1["current_s"].min()), float(df1["current_s"].max())),
        "temperature": (float(df1["temperature_s"].min()), float(df1["temperature_s"].max())),
        "friction_torque": (float(df1["friction_torque"].min()), float(df1["friction_torque"].max())),
    }
    delta_abs_all = pd.concat([df1["delta_I"].abs(), df2["delta_I"].abs()])
    ranges["abs_delta_I"] = (0.0, float(delta_abs_all.quantile(0.995)))

    df1["norm_current"] = minmax(df1["current_s"], *ranges["current"])
    df1["norm_temperature"] = minmax(df1["temperature_s"], *ranges["temperature"])
    df1["norm_F"] = minmax(df1["friction_torque"], *ranges["friction_torque"])
    df1["norm_abs_delta"] = minmax(df1["delta_I"].abs(), *ranges["abs_delta_I"])
    df1["HI1"] = 0.45 * df1["norm_current"] + 0.20 * df1["norm_temperature"] + 0.35 * df1["norm_F"]
    df1["HI2_like"] = (
        0.65 * df1["norm_current"] + 0.25 * df1["norm_temperature"] + 0.10 * df1["norm_abs_delta"]
    )

    df2["norm_current"] = minmax(df2["current_s"], *ranges["current"])
    df2["norm_temperature"] = minmax(df2["temperature_s"], *ranges["temperature"])
    df2["norm_abs_delta"] = minmax(df2["delta_I"].abs(), *ranges["abs_delta_I"])
    df2["HI2"] = (
        0.65 * df2["norm_current"] + 0.25 * df2["norm_temperature"] + 0.10 * df2["norm_abs_delta"]
    )
    df2["HI2_like"] = df2["HI2"]

    t1 = df1["day"].to_numpy(float)

    exp_fits = {}
    for name, col in [
        ("current_theoretical", "current_theoretical_s"),
        ("friction_torque", "friction_torque"),
        ("HI1", "HI1"),
    ]:
        popt, pcov, yhat, m = fit_exp(t1, df1[col].to_numpy(float))
        exp_fits[name] = {"params": popt, "pcov": pcov, "yhat": yhat, "metrics": m}

    piecewise = {}
    for name, col in [("current", "current_s"), ("HI1", "HI1")]:
        piecewise[name] = fit_continuous_piecewise(t1, df1[col].to_numpy(float), min_len=20)

    tau1 = piecewise["HI1"]["tau1"]
    tau2 = piecewise["HI1"]["tau2"]
    hi_tau1 = float(np.interp(tau1, df1["day"], df1["HI1"]))
    hi_tau2 = float(np.interp(tau2, df1["day"], df1["HI1"]))

    current = df2.iloc[-1]
    # Model inverse path uses the smoother mechanism-aligned theoretical current.
    t_inv = inverse_exp(float(current["current_theoretical_s"]), exp_fits["current_theoretical"]["params"])
    rul_inv = 3500 - t_inv

    # Residual bootstrap interval for inverse path.
    t_inv_ci = exp_bootstrap_inverse_ci(
        t1,
        df1["current_theoretical_s"].to_numpy(float),
        float(current["current_theoretical_s"]),
        exp_fits["current_theoretical"],
        n=1200,
    )
    rul_inv_ci = 3500 - t_inv_ci[[2, 1, 0]]

    daily_template = interpolate_daily(
        df1,
        ["norm_current", "norm_temperature", "HI2_like", "current_theoretical_s", "HI1"],
    )
    match_features = ["norm_current", "norm_temperature", "HI2_like"]
    match_results = []
    for window in [90, 180, 365]:
        match_results.append(
            sliding_template_match(
                daily_template,
                df2,
                feature_cols=match_features,
                window=window,
                weights=np.array([0.45, 0.25, 0.30]),
            )
        )
    primary_match = next(r for r in match_results if r["window"] == 180)
    t_match = primary_match["best_end_day"]
    rul_match = 3500 - t_match

    t_low = min(t_inv_ci[0], *(r["candidate_low"] for r in match_results), t_inv, t_match)
    t_high = max(t_inv_ci[2], *(r["candidate_high"] for r in match_results), t_inv, t_match)
    # Add half an attachment-1 sampling step as reporting uncertainty.
    t_low -= 10
    t_high += 10
    rul_interval = [3500 - t_high, 3500 - t_low]
    t_point = t_inv
    rul_point = 3500 - t_point

    def stage_from_t(t: float) -> str:
        if t < tau1:
            return "健康期"
        if t < tau2:
            return "退化期"
        return "衰退期"

    def trend_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        rows = []
        for col in cols:
            y = df[col].to_numpy(float)
            day = df["day"].to_numpy(float)
            rho = float(pd.DataFrame({"day": day, "value": y}).corr(method="spearman").iloc[0, 1])
            rows.append(
                {
                    "variable": col,
                    "start": float(y[0]),
                    "end": float(y[-1]),
                    "absolute_change": float(y[-1] - y[0]),
                    "relative_change_pct": float((y[-1] / y[0] - 1) * 100),
                    "linear_slope_per_day": float(np.polyfit(day, y, 1)[0]),
                    "spearman_rho": float(rho),
                }
            )
        return pd.DataFrame(rows)

    desc1 = df1[["day", "current", "current_theoretical", "temperature", "speed_rpm", "friction_torque", "delta_I"]].describe().T
    desc2 = df2[["day", "current", "current_theoretical", "temperature", "speed_rpm", "delta_I"]].describe().T
    trend1 = trend_table(df1, ["current", "current_theoretical", "temperature", "friction_torque", "delta_I"])
    trend2 = trend_table(df2, ["current", "current_theoretical", "temperature", "delta_I"])

    model_rows = []
    for name, fit in exp_fits.items():
        model_rows.append(
            {
                "model": "A_shifted_exp",
                "target": name,
                "c": fit["params"][0],
                "a": fit["params"][1],
                "b": fit["params"][2],
                **fit["metrics"],
            }
        )
    for name, fit in piecewise.items():
        model_rows.append(
            {
                "model": "B_three_stage_piecewise_linear",
                "target": name,
                "tau1": fit["tau1"],
                "tau2": fit["tau2"],
                "slope_1": fit["slopes"][0],
                "slope_2": fit["slopes"][1],
                "slope_3": fit["slopes"][2],
                **fit["metrics"],
            }
        )
    model_table = pd.DataFrame(model_rows)

    pseudo = pd.DataFrame(pseudo_online_predictions(df1, "current_theoretical_s", float(df1["current_theoretical_s"].iloc[-1])))

    match_summary = pd.DataFrame(
        [
            {
                "window_days": r["window"],
                "best_end_day": r["best_end_day"],
                "RUL": 3500 - r["best_end_day"],
                "best_score": r["best_score"],
                "candidate_low": r["candidate_low"],
                "candidate_high": r["candidate_high"],
                "RUL_interval_low": 3500 - r["candidate_high"],
                "RUL_interval_high": 3500 - r["candidate_low"],
                "candidate_count": r["candidate_count"],
            }
            for r in match_results
        ]
    )

    current_state = {
        "current_day_attachment2": float(current["day"]),
        "current_current_raw": float(current["current"]),
        "current_current_smooth": float(current["current_s"]),
        "current_current_theoretical_smooth": float(current["current_theoretical_s"]),
        "current_temperature_raw": float(current["temperature"]),
        "current_temperature_smooth": float(current["temperature_s"]),
        "current_delta_I": float(current["delta_I"]),
        "current_HI2": float(current["HI2"]),
        "tau1": tau1,
        "tau2": tau2,
        "HI1_at_tau1": hi_tau1,
        "HI1_at_tau2": hi_tau2,
        "t_inv": t_inv,
        "rul_inv": rul_inv,
        "t_inv_ci_95": t_inv_ci.tolist(),
        "rul_inv_ci_95": rul_inv_ci.tolist(),
        "t_match_180d": t_match,
        "rul_match_180d": rul_match,
        "rul_point": rul_point,
        "rul_interval_95_like": rul_interval,
        "stage_by_inverse": stage_from_t(t_inv),
        "stage_by_template": stage_from_t(t_match),
        "stage_conclusion": stage_from_t((t_inv + t_match) / 2),
    }

    desc1.to_csv(OUT / "attachment1_describe.csv")
    desc2.to_csv(OUT / "attachment2_describe.csv")
    trend1.to_csv(OUT / "attachment1_trends.csv", index=False)
    trend2.to_csv(OUT / "attachment2_trends.csv", index=False)
    model_table.to_csv(OUT / "model_fit_metrics.csv", index=False)
    pseudo.to_csv(OUT / "pseudo_online_exp_predictions.csv", index=False)
    match_summary.to_csv(OUT / "template_matching_summary.csv", index=False)
    df1.to_csv(OUT / "attachment1_processed.csv", index=False)
    df2.to_csv(OUT / "attachment2_processed.csv", index=False)
    with open(OUT / "current_state_and_rul.json", "w", encoding="utf-8") as f:
        json.dump(current_state, f, ensure_ascii=False, indent=2)

    # Figures as dependency-free SVGs.
    write_svg_plot(
        OUT / "fig1_attachment1_current_trends.svg",
        "Attachment 1 current trends",
        [
            {"x": df1["day"], "y": df1["current"], "label": "current raw", "color": "#9ecae1"},
            {"x": df1["day"], "y": df1["current_s"], "label": "current smoothed", "color": "#1f77b4"},
            {
                "x": df1["day"],
                "y": df1["current_theoretical_s"],
                "label": "theoretical smoothed",
                "color": "#ff7f0e",
            },
        ],
        y_label="Current (A)",
    )
    write_svg_plot(
        OUT / "fig1b_attachment1_temperature_friction.svg",
        "Attachment 1 temperature and friction (normalized view)",
        [
            {"x": df1["day"], "y": df1["norm_temperature"], "label": "temperature norm", "color": "#d62728"},
            {"x": df1["day"], "y": df1["norm_F"], "label": "friction norm", "color": "#2ca02c"},
        ],
        y_label="Normalized value",
    )
    write_svg_plot(
        OUT / "fig2_HI1_piecewise_stages.svg",
        "HI1 and three-stage piecewise fit",
        [
            {"x": df1["day"], "y": df1["HI1"], "label": "HI1", "color": "#1f77b4"},
            {
                "x": df1["day"],
                "y": piecewise["HI1"]["yhat"],
                "label": "piecewise linear fit",
                "color": "#d62728",
                "dash": True,
            },
        ],
        verticals=[
            {"x": tau1, "label": f"tau1={tau1:.0f}", "color": "#ff7f0e"},
            {"x": tau2, "label": f"tau2={tau2:.0f}", "color": "#d62728"},
        ],
        y_label="HI1",
    )
    write_svg_plot(
        OUT / "fig3_exp_fit_current_theoretical.svg",
        "Shifted exponential fit on theoretical current",
        [
            {
                "x": df1["day"],
                "y": df1["current_theoretical_s"],
                "label": "theoretical smoothed",
                "color": "#1f77b4",
            },
            {
                "x": df1["day"],
                "y": exp_fits["current_theoretical"]["yhat"],
                "label": "shifted exponential",
                "color": "#d62728",
                "dash": True,
            },
        ],
        y_label="Current (A)",
    )
    write_svg_plot(
        OUT / "fig4_attachment2_equivalent_life.svg",
        "Attachment 2 equivalent life position",
        [
            {"x": df1["day"], "y": df1["HI1"], "label": "Attachment 1 HI1", "color": "#1f77b4"},
            {"x": df2["day"], "y": df2["HI2"], "label": "Attachment 2 HI2", "color": "#2ca02c"},
        ],
        verticals=[
            {"x": t_inv, "label": f"inverse={t_inv:.0f}", "color": "#9467bd"},
            {"x": t_match, "label": f"match={t_match:.0f}", "color": "#2ca02c"},
            {"x": tau1, "label": f"tau1={tau1:.0f}", "color": "#ff7f0e"},
            {"x": tau2, "label": f"tau2={tau2:.0f}", "color": "#d62728"},
        ],
        y_label="Health index",
    )

    print(json.dumps(current_state, ensure_ascii=False, indent=2))
    print("\nMODEL TABLE")
    print(model_table.to_string(index=False))
    print("\nMATCH SUMMARY")
    print(match_summary.to_string(index=False))
    print("\nPSEUDO ONLINE")
    print(pseudo.to_string(index=False))


if __name__ == "__main__":
    main()
