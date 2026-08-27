"""
発展分析: avg_active_nodes版のevaluation.py並行コピー。
元のevaluation.py(power_consumption対象, results/eval/)は変更しない。
出力先はresults/eval_nodes/に分離。changepoints.csv(レジーム境界)は元の富岳全体分析と共有。
"""
import os

import numpy as np
import pandas as pd

from features_nodes import build_feature_table, load_split_ranges, TARGET_COL

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval_nodes")
os.makedirs(EVAL_DIR, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TRANSITION_WINDOW_DAYS = 14


def naive_seasonal_mae_scale(period_hours=24):
    table = build_feature_table()
    ranges = load_split_ranges()
    start, end = ranges["train"]
    y = table.loc[(table.index >= start) & (table.index < end), TARGET_COL]
    naive_err = (y - y.shift(period_hours)).abs().dropna()
    return float(naive_err.mean())


def load_changepoints():
    cp = pd.read_csv(os.path.join(RESULTS, "changepoints.csv"), parse_dates=["changepoint_date"])
    dt = cp["changepoint_date"].dt
    if cp["changepoint_date"].dt.tz is None:
        return dt.tz_localize("Asia/Tokyo").tolist()
    return dt.tz_convert("Asia/Tokyo").tolist()


def label_regime(target_times, changepoints, window_days=TRANSITION_WINDOW_DAYS):
    target_times = pd.DatetimeIndex(target_times)
    window = pd.Timedelta(days=window_days)
    is_transition = np.zeros(len(target_times), dtype=bool)
    for cp in changepoints:
        is_transition |= (target_times >= cp - window) & (target_times <= cp + window)
    return np.where(is_transition, "transition", "stable")


def horizon_bucket(horizons):
    horizons = np.asarray(horizons)
    return np.where(horizons <= 24, "short_1-24h", "medium_48-168h")


def evaluate(model_name, info_level, meta, y_true, quantile_preds, naive_scale=None, save=True):
    y_true = np.asarray(y_true, dtype=np.float64)
    if naive_scale is None:
        naive_scale = naive_seasonal_mae_scale()

    changepoints = load_changepoints()
    regime = label_regime(meta["target_time"].to_numpy(), changepoints)
    hbucket = horizon_bucket(meta["horizon"].to_numpy())

    median = np.asarray(quantile_preds[0.5], dtype=np.float64)
    abs_err = np.abs(y_true - median)
    sq_err = (y_true - median) ** 2

    per_row_pinball = np.zeros(len(y_true))
    for q in QUANTILES:
        pred_q = np.asarray(quantile_preds[q], dtype=np.float64)
        diff = y_true - pred_q
        per_row_pinball += np.maximum(q * diff, (q - 1) * diff)
    per_row_pinball /= len(QUANTILES)
    per_row_crps_approx = 2 * per_row_pinball

    raw = pd.DataFrame({
        "origin": meta["origin"].to_numpy(),
        "horizon": meta["horizon"].to_numpy(),
        "target_time": meta["target_time"].to_numpy(),
        "y_true": y_true,
        "y_pred_median": median,
        "abs_err": abs_err,
        "sq_err": sq_err,
        "pinball": per_row_pinball,
        "crps_approx": per_row_crps_approx,
        "horizon_bucket": hbucket,
        "regime": regime,
    })
    for q in QUANTILES:
        raw[f"y_pred_q{q}"] = np.asarray(quantile_preds[q], dtype=np.float64)

    groups = [("overall", "overall", raw)]
    for hb in ["short_1-24h", "medium_48-168h"]:
        for rg in ["stable", "transition"]:
            sub = raw[(raw["horizon_bucket"] == hb) & (raw["regime"] == rg)]
            if len(sub) > 0:
                groups.append((hb, rg, sub))

    summary_rows = []
    for hb, rg, sub in groups:
        summary_rows.append({
            "model": model_name, "info_level": info_level,
            "horizon_bucket": hb, "regime": rg, "n": len(sub),
            "MAE": sub["abs_err"].mean(),
            "RMSE": np.sqrt(sub["sq_err"].mean()),
            "MASE": sub["abs_err"].mean() / naive_scale,
            "pinball_avg": sub["pinball"].mean(),
            "CRPS_approx": sub["crps_approx"].mean(),
        })
    summary = pd.DataFrame(summary_rows)

    if save:
        raw.to_parquet(os.path.join(EVAL_DIR, f"{model_name}_raw.parquet"), index=False)
        summary.to_csv(os.path.join(EVAL_DIR, f"{model_name}_summary.csv"), index=False)
    return summary, raw
