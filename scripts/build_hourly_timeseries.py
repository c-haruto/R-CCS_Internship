"""
Step 1: ジョブ単位ログ -> 1時間グリッド多変量時系列の構築

各ジョブの実行区間 [sdt, edt) を1時間バケットへ按分(overlap prorate)し、
- 消費電力 (econ を按分)
- 稼働ノード数 / 同時実行ジョブ数 (node-seconds, job-seconds を按分)
- compute-bound / memory-bound 比率
- 平均FLOPS・メモリ帯域幅 (時間按分の重み付き平均)
を計算する。加えて adt (投入時刻) 基準で
- ジョブ投入率(到着数)、リクエストノード数/コア数、平均待ち時間、新規ユーザ出現数
を、edt (終了時刻) 基準で
- 完了ジョブ数、失敗率
を集計する。

出力: results/hourly_timeseries.parquet (時間グリッド x 多変量特徴量)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

DATA_DIR = "/home/elicamodd/f-data"
OUT_DIR = os.path.expanduser("~/projects/fugaku-power-forecast/results")
os.makedirs(OUT_DIR, exist_ok=True)

# データ全体の時刻範囲（事前スキャン済み: sdt min 2021-03-01 00:20, edt max 2024-05-14 02:38 JST）
GRID_START = pd.Timestamp("2021-03-01 00:00:00", tz="Asia/Tokyo")
GRID_END = pd.Timestamp("2024-05-14 03:00:00", tz="Asia/Tokyo")
N_HOURS = int((GRID_END - GRID_START) / pd.Timedelta(hours=1)) + 1
GRID_START_SEC = GRID_START.value // 10**9

COLS = [
    "usr", "cnumr", "cnumat", "nnumr", "adt", "sdt", "edt", "ec", "nnuma",
    "econ", "avgpcon", "flops", "mbwidth", "pclass", "exit state", "duration",
]


def new_accumulators(n):
    keys = [
        "active_node_seconds", "concurrent_job_seconds", "active_core_seconds",
        "energy_prorated", "compute_bound_node_seconds", "memory_bound_node_seconds",
        "flops_weighted", "mbwidth_weighted", "perf_weight_seconds",
        "job_arrivals", "req_nodes_sum", "req_cores_sum",
        "wait_time_sum", "wait_time_count", "new_user_count",
        "jobs_completed", "jobs_failed",
    ]
    return {k: np.zeros(n, dtype=np.float64) for k in keys}


def add_at(arr, idx, values, valid_mask=None):
    if valid_mask is not None:
        idx = idx[valid_mask]
        values = values[valid_mask]
    if len(idx) == 0:
        return
    np.add.at(arr, idx, values)


def process_file(path, acc, seen_users):
    df = pd.read_parquet(path, columns=COLS)
    n_raw = len(df)

    for c in ["adt", "sdt", "edt"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df = df[df["sdt"].notna() & df["edt"].notna() & df["adt"].notna()]
    df = df[df["edt"] >= df["sdt"]]
    df = df[df["econ"].abs() < 1e10]  # drop corrupted extreme values
    df = df.sort_values("adt").reset_index(drop=True)
    n_kept = len(df)

    sdt_sec = df["sdt"].values.astype("datetime64[s]").astype(np.int64)
    edt_sec = df["edt"].values.astype("datetime64[s]").astype(np.int64)
    adt_sec = df["adt"].values.astype("datetime64[s]").astype(np.int64)

    nnuma = df["nnuma"].to_numpy(dtype=np.float64)
    cnumat = df["cnumat"].to_numpy(dtype=np.float64)
    econ = df["econ"].to_numpy(dtype=np.float64)
    flops = df["flops"].to_numpy(dtype=np.float64)
    mbwidth = df["mbwidth"].to_numpy(dtype=np.float64)
    pclass = df["pclass"].to_numpy()

    # ---------- Arrival-based (adt bucket) ----------
    arr_idx = (adt_sec - GRID_START_SEC) // 3600
    valid_arr = (arr_idx >= 0) & (arr_idx < N_HOURS)
    arr_idx_v = arr_idx.astype(np.int64)

    add_at(acc["job_arrivals"], arr_idx_v, np.ones(n_kept), valid_arr)
    add_at(acc["req_nodes_sum"], arr_idx_v, df["nnumr"].to_numpy(dtype=np.float64), valid_arr)
    add_at(acc["req_cores_sum"], arr_idx_v, df["cnumr"].to_numpy(dtype=np.float64), valid_arr)

    wait_sec = (sdt_sec - adt_sec).astype(np.float64)
    wait_valid = valid_arr & (wait_sec >= 0)
    add_at(acc["wait_time_sum"], arr_idx_v, wait_sec, wait_valid)
    add_at(acc["wait_time_count"], arr_idx_v, np.ones(n_kept), wait_valid)

    # new-user detection: first occurrence (by adt order) not seen in any prior file
    first_in_file = ~df["usr"].duplicated(keep="first")
    is_new = first_in_file.to_numpy() & (~df["usr"].isin(seen_users).to_numpy())
    add_at(acc["new_user_count"], arr_idx_v, np.ones(n_kept), valid_arr & is_new)
    seen_users.update(df["usr"].unique().tolist())

    # ---------- Completion-based (edt bucket) ----------
    end_idx = (edt_sec - GRID_START_SEC) // 3600
    valid_end = (end_idx >= 0) & (end_idx < N_HOURS)
    end_idx_v = end_idx.astype(np.int64)
    is_failed = (df["exit state"] == "failed").to_numpy().astype(np.float64)

    add_at(acc["jobs_completed"], end_idx_v, np.ones(n_kept), valid_end)
    add_at(acc["jobs_failed"], end_idx_v, is_failed, valid_end)

    # ---------- Interval overlap (sdt -> edt, hourly prorate) ----------
    edt_eff = np.maximum(edt_sec, sdt_sec + 1)
    dur = (edt_eff - sdt_sec).astype(np.float64)  # >=1 sec

    bucket_first = (sdt_sec // 3600) * 3600
    bucket_last = ((edt_eff - 1) // 3600) * 3600
    n_buckets = ((bucket_last - bucket_first) // 3600 + 1).astype(np.int64)

    compute_mask = pclass == "compute-bound"
    memory_mask = pclass == "memory-bound"
    perf_valid = np.isfinite(flops) & np.isfinite(mbwidth) & (flops >= 0) & (mbwidth >= 0)

    def accumulate_overlap(idx_bucket_start, overlap_sec, nnuma_v, cnumat_v, econ_v, dur_v,
                            compute_v, memory_v, flops_v, mbwidth_v, perf_valid_v):
        idx = ((idx_bucket_start - GRID_START_SEC) // 3600).astype(np.int64)
        valid = (idx >= 0) & (idx < N_HOURS) & (overlap_sec > 0)

        node_sec = nnuma_v * overlap_sec
        add_at(acc["active_node_seconds"], idx, node_sec, valid)
        add_at(acc["active_core_seconds"], idx, cnumat_v * overlap_sec, valid)
        add_at(acc["concurrent_job_seconds"], idx, overlap_sec, valid)
        add_at(acc["energy_prorated"], idx, econ_v * (overlap_sec / dur_v), valid)
        add_at(acc["compute_bound_node_seconds"], idx, node_sec, valid & compute_v)
        add_at(acc["memory_bound_node_seconds"], idx, node_sec, valid & memory_v)

        pv = valid & perf_valid_v
        add_at(acc["flops_weighted"], idx, flops_v * overlap_sec, pv)
        add_at(acc["mbwidth_weighted"], idx, mbwidth_v * overlap_sec, pv)
        add_at(acc["perf_weight_seconds"], idx, overlap_sec, pv)

    # --- fast path: jobs contained in a single hour bucket ---
    mask1 = n_buckets == 1
    if mask1.any():
        overlap1 = dur[mask1]
        accumulate_overlap(
            bucket_first[mask1], overlap1,
            nnuma[mask1], cnumat[mask1], econ[mask1], dur[mask1],
            compute_mask[mask1], memory_mask[mask1], flops[mask1], mbwidth[mask1], perf_valid[mask1],
        )

    # --- slow path: jobs spanning multiple hour buckets ---
    mask2 = n_buckets > 1
    if mask2.any():
        nb2 = n_buckets[mask2]
        starts = np.cumsum(nb2) - nb2
        total = int(nb2.sum())
        offsets = np.arange(total) - np.repeat(starts, nb2)

        bucket_rep = np.repeat(bucket_first[mask2], nb2) + offsets * 3600
        bucket_end_rep = bucket_rep + 3600
        sdt_rep = np.repeat(sdt_sec[mask2], nb2)
        edt_rep = np.repeat(edt_eff[mask2], nb2)
        overlap2 = np.clip(
            np.minimum(edt_rep, bucket_end_rep) - np.maximum(sdt_rep, bucket_rep), 0, 3600
        ).astype(np.float64)

        accumulate_overlap(
            bucket_rep, overlap2,
            np.repeat(nnuma[mask2], nb2), np.repeat(cnumat[mask2], nb2),
            np.repeat(econ[mask2], nb2), np.repeat(dur[mask2], nb2),
            np.repeat(compute_mask[mask2], nb2), np.repeat(memory_mask[mask2], nb2),
            np.repeat(flops[mask2], nb2), np.repeat(mbwidth[mask2], nb2), np.repeat(perf_valid[mask2], nb2),
        )

    return n_raw, n_kept, int(mask2.sum()) if mask2.any() else 0


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"{len(files)} files, N_HOURS={N_HOURS} ({GRID_START} .. {GRID_END})")

    acc = new_accumulators(N_HOURS)
    seen_users = set()

    for i, f in enumerate(files):
        t1 = time.time()
        n_raw, n_kept, n_multi = process_file(f, acc, seen_users)
        print(
            f"[{i + 1}/{len(files)}] {os.path.basename(f)}: "
            f"raw={n_raw:,} kept={n_kept:,} multi_bucket_jobs={n_multi:,} "
            f"({time.time() - t1:.1f}s)"
        )

    hour_index = pd.date_range(GRID_START, periods=N_HOURS, freq="h")
    out = pd.DataFrame({"timestamp": hour_index})

    def safe_div(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)

    out["power_consumption"] = acc["energy_prorated"]  # 主ターゲット: 時間別の総消費電力(prorated econ)
    out["avg_active_nodes"] = acc["active_node_seconds"] / 3600.0
    out["avg_active_cores"] = acc["active_core_seconds"] / 3600.0
    out["avg_concurrent_jobs"] = acc["concurrent_job_seconds"] / 3600.0
    out["compute_bound_ratio"] = safe_div(
        acc["compute_bound_node_seconds"],
        acc["compute_bound_node_seconds"] + acc["memory_bound_node_seconds"],
    )
    out["avg_flops"] = safe_div(acc["flops_weighted"], acc["perf_weight_seconds"])
    out["avg_mbwidth"] = safe_div(acc["mbwidth_weighted"], acc["perf_weight_seconds"])

    out["job_arrival_count"] = acc["job_arrivals"]
    out["avg_requested_nodes"] = safe_div(acc["req_nodes_sum"], acc["job_arrivals"])
    out["avg_requested_cores"] = safe_div(acc["req_cores_sum"], acc["job_arrivals"])
    out["avg_wait_time_sec"] = safe_div(acc["wait_time_sum"], acc["wait_time_count"])
    out["new_user_count"] = acc["new_user_count"]

    out["jobs_completed_count"] = acc["jobs_completed"]
    out["failure_rate"] = safe_div(acc["jobs_failed"], acc["jobs_completed"])

    out_path = os.path.join(OUT_DIR, "hourly_timeseries.parquet")
    out.to_parquet(out_path, index=False)
    csv_path = os.path.join(OUT_DIR, "hourly_timeseries.csv")
    out.to_csv(csv_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {csv_path}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"\nUnique users seen total: {len(seen_users):,}")
    print("\n--- describe ---")
    print(out.describe())


if __name__ == "__main__":
    main()
