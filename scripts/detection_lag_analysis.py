"""
Step 3: 検知遅延・性能回復時間の分析

各変化点(results/test_changepoints.csv)の前後で、horizon=1h(最も直近の情報を使う=最も
反応が速いはずの)予測誤差がどう推移するかを見て、
- 遷移直後にどれだけ誤差が跳ね上がるか(ピーク)
- 誤差がベースライン水準(遷移前14〜3日平均の1.3倍以内)に戻るまで何日かかるか(回復日数)
をモデルごとに算出する。②power-onlyと③+covariateの回復速度差が、
「covariateがレジーム遷移への適応を速めるか」という問いに対するより直接的な答えになる。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval")

MODEL_FILES = {
    "chronos_bolt_small": ("①Chronos", "zero-shot"),
    "lightgbm_info2": ("②LightGBM", "power-only"),
    "lightgbm_info3": ("③LightGBM", "+covariate"),
    "lstm_info2": ("②LSTM", "power-only"),
    "lstm_info3": ("③LSTM", "+covariate"),
    "tft_info2": ("②TFT-lite", "power-only"),
    "tft_info3": ("③TFT-lite", "+covariate"),
}
WINDOW_DAYS = 30
BASELINE_RANGE = (-14, -3)
RECOVERY_THRESHOLD_MULT = 1.3
RECOVERY_SUSTAIN_DAYS = 3


def load_changepoints():
    cp = pd.read_csv(os.path.join(RESULTS, "test_changepoints.csv"), parse_dates=["changepoint_date"])
    dt = cp["changepoint_date"].dt
    if cp["changepoint_date"].dt.tz is None:
        return dt.tz_localize("Asia/Tokyo").tolist()
    return dt.tz_convert("Asia/Tokyo").tolist()


def daily_h1_error(model_name):
    raw = pd.read_parquet(os.path.join(EVAL_DIR, f"{model_name}_raw.parquet"))
    h1 = raw[raw["horizon"] == 1].copy()
    h1["date"] = pd.to_datetime(h1["target_time"]).dt.normalize()
    return h1.groupby("date")["abs_err"].mean()


def recovery_stats(series, cp, window_days=WINDOW_DAYS):
    cp_date = pd.Timestamp(cp).normalize()
    offsets = (series.index - cp_date).days
    mask = (offsets >= -window_days) & (offsets <= window_days)
    sub = series[mask]
    sub_offsets = offsets[mask]
    curve = pd.Series(sub.values, index=sub_offsets).sort_index()

    base_mask = (curve.index >= BASELINE_RANGE[0]) & (curve.index <= BASELINE_RANGE[1])
    if base_mask.sum() < 3:
        return None
    baseline = curve[base_mask].mean()
    threshold = baseline * RECOVERY_THRESHOLD_MULT

    post = curve[curve.index >= 0]
    if len(post) == 0:
        return None
    peak = post.max()
    peak_offset = post.idxmax()

    roll = post.rolling(RECOVERY_SUSTAIN_DAYS, min_periods=RECOVERY_SUSTAIN_DAYS).mean()
    recovered = roll[roll <= threshold]
    recovery_offset = int(recovered.index[0]) if len(recovered) > 0 else np.nan

    return {
        "baseline": baseline, "threshold": threshold,
        "peak": peak, "peak_offset": int(peak_offset),
        "recovery_offset": recovery_offset,
        "curve": curve,
    }


def main():
    changepoints = load_changepoints()
    series_by_model = {m: daily_h1_error(m) for m in MODEL_FILES}

    rows = []
    curves = {}  # (model, cp_idx) -> curve series, for plotting
    for m, (label, variant) in MODEL_FILES.items():
        for ci, cp in enumerate(changepoints):
            stats = recovery_stats(series_by_model[m], cp)
            if stats is None:
                continue
            rows.append({
                "model": m, "label": label, "variant": variant,
                "changepoint": cp.date(), "changepoint_idx": ci,
                "baseline_h1_mae": stats["baseline"], "peak_h1_mae": stats["peak"],
                "peak_offset_days": stats["peak_offset"],
                "recovery_offset_days": stats["recovery_offset"],
                "peak_to_baseline_ratio": stats["peak"] / stats["baseline"],
            })
            curves[(m, ci)] = stats["curve"]

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, "step3_detection_lag.csv")
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}\n")

    summary = df.groupby(["label", "variant"]).agg(
        mean_peak_to_baseline_ratio=("peak_to_baseline_ratio", "mean"),
        mean_recovery_offset_days=("recovery_offset_days", "mean"),
        pct_recovered_within_30d=("recovery_offset_days", lambda s: s.notna().mean() * 100),
        n_changepoints=("changepoint", "count"),
    ).round(2)
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS, "step3_detection_lag_summary.csv"))

    # --- plot: per changepoint, ②vs③ curves per architecture (Chronos as reference) ---
    archs = [("LightGBM", "lightgbm_info2", "lightgbm_info3"),
             ("LSTM", "lstm_info2", "lstm_info3"),
             ("TFT-lite", "tft_info2", "tft_info3")]
    n_cp = len(changepoints)
    fig, axes = plt.subplots(len(archs), n_cp, figsize=(4 * n_cp, 3.2 * len(archs)), sharex=True)
    for ai, (arch_name, m2, m3) in enumerate(archs):
        for ci, cp in enumerate(changepoints):
            ax = axes[ai, ci]
            if (m2, ci) in curves:
                c2 = curves[(m2, ci)]
                ax.plot(c2.index, c2.values, color="#4c78a8", label="②power-only", lw=1.2)
            if (m3, ci) in curves:
                c3 = curves[(m3, ci)]
                ax.plot(c3.index, c3.values, color="#f58518", label="③+covariate", lw=1.2)
            if ("chronos_bolt_small", ci) in curves:
                cc = curves[("chronos_bolt_small", ci)]
                ax.plot(cc.index, cc.values, color="gray", label="①Chronos", lw=1.0, ls=":")
            ax.axvline(0, color="black", lw=0.8)
            if ai == 0:
                ax.set_title(str(cp.date()))
            if ci == 0:
                ax.set_ylabel(f"{arch_name}\nh=1 abs_err")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    plt.suptitle("Detection-lag: horizon=1h forecast error around each test-period changepoint (day 0 = changepoint)")
    plt.tight_layout()
    fig_path = os.path.join(RESULTS, "step3_detection_lag_curves.png")
    plt.savefig(fig_path, dpi=100)
    print(f"\nsaved: {fig_path}")


if __name__ == "__main__":
    main()
