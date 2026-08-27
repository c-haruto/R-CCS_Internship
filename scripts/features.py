"""
Step 2 共通モジュール: 特徴量エンジニアリング

- 電力(power_consumption)由来の特徴量のみ (モデル②用): ラグ・rolling統計・カレンダー特徴量
- 上記 + ジョブログ由来covariate (モデル③用): job_arrival_count, avg_requested_nodes/cores,
  new_user_count, avg_wait_time_sec, compute_bound_ratio, failure_rate のラグ・rolling・急増z-score

方針:
- 最大参照コンテキストは 720時間(30日)に統一（Chronosとの公平性; README 2.3参照）。
- direct multi-horizon方式: 各予測起点(origin)の特徴量 + horizon(整数, 1-168h) + 予測対象時刻の
  カレンダー特徴量 を入力とし、origin+horizon時点のpower_consumptionを目的変数とする1本のモデルで
  短期(1-24h, 毎時)・中期(24-168h, 24h刻み)の両方をカバーする。
- train/valid/testを跨ぐ(origin, horizon)サンプルはリークになるため生成しない
  (originとorigin+horizonが同一splitに収まるものだけを採用)。
"""
import json
import os

import numpy as np
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
MAX_LOOKBACK = 720  # hours, Chronosとの公平性のための上限 (README 2.3)

POWER_LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720]
POWER_ROLL_WINDOWS = [24, 72, 168, 336, 720]

COVARIATE_COLS = [
    "job_arrival_count", "avg_requested_nodes", "avg_requested_cores",
    "new_user_count", "avg_wait_time_sec", "compute_bound_ratio", "failure_rate",
]
# job_arrival_count/new_user_countは真のゼロ件で欠損しない。それ以外はその時間内に
# 分母となるイベント(到着 or 完了)が0件だと未定義(NaN)になるため、直前の観測値で
# 前方補完する（「新規到着が無い間は直近の傾向が続く」とみなす）。
COV_COLS_NEED_FFILL = ["avg_requested_nodes", "avg_requested_cores", "avg_wait_time_sec",
                        "compute_bound_ratio", "failure_rate"]
COV_LAGS = [0, 1, 24, 168]
COV_ROLL_WINDOWS = [24, 168]
COV_Z_WINDOWS = [168, 720]  # 急増シグナル (rolling z-score)

# 短期(毎時 1-24h) + 中期(24h刻み 48-168h)
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
    base = series.shift(1)  # 現在値自身を基準統計から除外(リーク防止)
    for w in windows:
        min_p = max(12, w // 4)
        mean = base.rolling(w, min_periods=min_p).mean()
        std = base.rolling(w, min_periods=min_p).std().replace(0, np.nan)
        out[f"{prefix}_z{w}h"] = (series - mean) / std
    return out


def build_feature_table():
    """timestampをindexに持つ、全時刻共通の特徴量テーブルを構築する。"""
    df = load_hourly()
    assert df.index.is_monotonic_increasing
    df = df.copy()
    df[COV_COLS_NEED_FFILL] = df[COV_COLS_NEED_FFILL].ffill().bfill()

    feats = [df[["power_consumption"]]]
    feats.append(_lag_roll_features(df["power_consumption"], POWER_LAGS, POWER_ROLL_WINDOWS, "power"))
    feats.append(_calendar_features(df.index))

    for col in COVARIATE_COLS:
        feats.append(_lag_roll_features(df[col], COV_LAGS, COV_ROLL_WINDOWS, col))
        feats.append(_zscore_features(df[col], COV_Z_WINDOWS, col))

    table = pd.concat(feats, axis=1)

    # MAX_LOOKBACKを超えるラグ/rolling列が紛れ込んでいないか検証
    for lag in POWER_LAGS + COV_LAGS:
        assert lag <= MAX_LOOKBACK
    for w in POWER_ROLL_WINDOWS + COV_ROLL_WINDOWS + COV_Z_WINDOWS:
        assert w <= MAX_LOOKBACK

    return table


def power_only_columns(table):
    return [c for c in table.columns if c.startswith("power_") or c.startswith("cal_")]


def covariate_columns(table):
    cols = []
    for col in COVARIATE_COLS:
        cols += [c for c in table.columns if c.startswith(col + "_")]
    return cols


def make_direct_horizon_samples(table, split_name, info_level, origin_step=6):
    """
    direct multi-horizon 方式のtabularサンプルを生成する。

    split_name: 'train' | 'valid' | 'test'
    info_level: '2' (power-only, モデル②) | '3' (power+covariate, モデル③)
    origin_step: 起点を何時間おきに取るか（train/validは密に, testは評価harness側で別途制御）

    戻り値: X (DataFrame), y (Series), meta (DataFrame: origin, horizon, target_time)
    """
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
        cal_t.index = o  # origin基準でindexを揃えてconcatできるようにする
        cal_t = cal_t.add_prefix("target_")
        X_h = pd.concat([X_h.reset_index(drop=True), cal_t.reset_index(drop=True)], axis=1)
        y_h = table.loc[t, "power_consumption"].to_numpy()

        rows.append(X_h)
        metas.append(pd.DataFrame({"origin": o, "horizon": h, "target_time": t}))

    X = pd.concat(rows, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    y = pd.Series(table.loc[pd.DatetimeIndex(meta["target_time"]), "power_consumption"].to_numpy(), name="y")

    # 学習に使える行のみ残す（最大ラグ分の履歴が無い先頭付近はNaNが残る）
    valid_rows = X.notna().all(axis=1) & y.notna()
    return X.loc[valid_rows].reset_index(drop=True), y.loc[valid_rows].reset_index(drop=True), meta.loc[valid_rows].reset_index(drop=True)


if __name__ == "__main__":
    table = build_feature_table()
    print("feature table shape:", table.shape)
    print("power-only feature cols:", len(power_only_columns(table)))
    print("covariate feature cols:", len(covariate_columns(table)))

    X, y, meta = make_direct_horizon_samples(table, "train", "3", origin_step=6)
    print("\ntrain (info_level=3) samples:", X.shape, y.shape)
    print(meta.head())
    print("\nNaN check:", X.isna().sum().sum(), "y NaN:", y.isna().sum())
