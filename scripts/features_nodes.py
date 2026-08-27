"""
発展分析: avg_active_nodesを予測対象としたfeatures.pyの並行版

元のfeatures.py(power_consumption対象)は変更せず、target_colをavg_active_nodesに
差し替えた独立コピー。covariateの集合(COVARIATE_COLS)は元と同一(7種)に揃え、
「target系列を変えても同じ結論が出るか」を検証する対照実験として使う。
"""
import json
import os

import numpy as np
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
MAX_LOOKBACK = 720

TARGET_COL = "avg_active_nodes"

TARGET_LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720]
TARGET_ROLL_WINDOWS = [24, 72, 168, 336, 720]

COVARIATE_COLS = [
    "job_arrival_count", "avg_requested_nodes", "avg_requested_cores",
    "new_user_count", "avg_wait_time_sec", "compute_bound_ratio", "failure_rate",
]
COV_COLS_NEED_FFILL = ["avg_requested_nodes", "avg_requested_cores", "avg_wait_time_sec",
                        "compute_bound_ratio", "failure_rate"]
COV_LAGS = [0, 1, 24, 168]
COV_ROLL_WINDOWS = [24, 168]
COV_Z_WINDOWS = [168, 720]

HORIZONS = list(range(1, 25)) + list(range(48, 169, 24))


def load_hourly():
    df = pd.read_parquet(os.path.join(RESULTS, "hourly_timeseries.parquet"))
    df = df.set_index("timestamp").sort_index()
    return df


def load_split_ranges():
    with open(os.path.join(RESULTS, "split_design.json")) as f:
        raw = json.load(f)
    tz = "Asia/Tokyo"
    ranges = {}
    for name in ["train", "valid", "test"]:
        ranges[name] = (pd.Timestamp(raw[name]["start"], tz=tz), pd.Timestamp(raw[name]["end"], tz=tz))
    return ranges


def _calendar_features(index):
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    cal = pd.DataFrame(index=index)
    cal["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cal["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    cal["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    cal["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    cal["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["cal_is_weekend"] = (dow >= 5).astype(np.float64)
    return cal


def _lag_roll_features(series, lags, roll_windows, prefix):
    out = pd.DataFrame(index=series.index)
    for lag in lags:
        out[f"{prefix}_lag{lag}h"] = series.shift(lag)
    for w in roll_windows:
        min_p = max(6, w // 4)
        out[f"{prefix}_rollmean{w}h"] = series.rolling(w, min_periods=min_p).mean()
        out[f"{prefix}_rollstd{w}h"] = series.rolling(w, min_periods=min_p).std()
    return out


def _zscore_features(series, windows, prefix):
    out = pd.DataFrame(index=series.index)
    base = series.shift(1)
    for w in windows:
        min_p = max(12, w // 4)
        mean = base.rolling(w, min_periods=min_p).mean()
        std = base.rolling(w, min_periods=min_p).std().replace(0, np.nan)
        out[f"{prefix}_z{w}h"] = (series - mean) / std
    return out


def build_feature_table():
    df = load_hourly()
    assert df.index.is_monotonic_increasing
    df = df.copy()
    df[COV_COLS_NEED_FFILL] = df[COV_COLS_NEED_FFILL].ffill().bfill()

    feats = [df[[TARGET_COL]]]
    feats.append(_lag_roll_features(df[TARGET_COL], TARGET_LAGS, TARGET_ROLL_WINDOWS, "target"))
    feats.append(_calendar_features(df.index))

    for col in COVARIATE_COLS:
        feats.append(_lag_roll_features(df[col], COV_LAGS, COV_ROLL_WINDOWS, col))
        feats.append(_zscore_features(df[col], COV_Z_WINDOWS, col))

    table = pd.concat(feats, axis=1)

    for lag in TARGET_LAGS + COV_LAGS:
        assert lag <= MAX_LOOKBACK
    for w in TARGET_ROLL_WINDOWS + COV_ROLL_WINDOWS + COV_Z_WINDOWS:
        assert w <= MAX_LOOKBACK

    return table


def power_only_columns(table):
    return [c for c in table.columns if c.startswith("target_") or c.startswith("cal_")]


def covariate_columns(table):
    cols = []
    for col in COVARIATE_COLS:
        cols += [c for c in table.columns if c.startswith(col + "_")]
    return cols


def make_direct_horizon_samples(table, split_name, info_level, origin_step=6):
    ranges = load_split_ranges()
    start, end = ranges[split_name]

    feature_cols = power_only_columns(table)
    if info_level == "3":
        feature_cols = feature_cols + covariate_columns(table)

    origins = table.index[(table.index >= start) & (table.index < end)]
    origins = origins[::origin_step]

    rows = []
    metas = []
    for h in HORIZONS:
        target_times = origins + pd.Timedelta(hours=h)
        valid_mask = (target_times >= start) & (target_times < end) & target_times.isin(table.index)
        o = origins[valid_mask]
        t = target_times[valid_mask]
        if len(o) == 0:
            continue
        X_h = table.loc[o, feature_cols].copy()
        X_h["horizon"] = h
        cal_t = _calendar_features(t)
        cal_t.index = o
        cal_t = cal_t.add_prefix("target_")
        X_h = pd.concat([X_h.reset_index(drop=True), cal_t.reset_index(drop=True)], axis=1)

        rows.append(X_h)
        metas.append(pd.DataFrame({"origin": o, "horizon": h, "target_time": t}))

    X = pd.concat(rows, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    y = pd.Series(table.loc[pd.DatetimeIndex(meta["target_time"]), TARGET_COL].to_numpy(), name="y")

    valid_rows = X.notna().all(axis=1) & y.notna()
    return X.loc[valid_rows].reset_index(drop=True), y.loc[valid_rows].reset_index(drop=True), meta.loc[valid_rows].reset_index(drop=True)
