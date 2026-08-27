"""
Step 3: results/test_changepoints.csv (test期間内で検出した5変化点) を使って、
既存の7モデルの生予測(results/eval/*_raw.parquet)の regime ラベル(安定期/遷移期)を
再付与し、集計指標を再計算する。予測自体の再計算は不要(モデルは凍結済み)。
"""
import os

import numpy as np
import pandas as pd

from evaluation import naive_seasonal_mae_scale, QUANTILES, TRANSITION_WINDOW_DAYS

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval")

MODEL_FILES = {
    "chronos_bolt_small": ("1", "①zero-shot", "Chronos-bolt (zero-shot)"),
    "lightgbm_info2": ("2", "②power-only", "LightGBM"),
    "lightgbm_info3": ("3", "③+covariate", "LightGBM"),
    "lstm_info2": ("2", "②power-only", "LSTM"),
    "lstm_info3": ("3", "③+covariate", "LSTM"),
    "tft_info2": ("2", "②power-only", "TFT-lite"),
    "tft_info3": ("3", "③+covariate", "TFT-lite"),
}


def load_test_changepoints():
    cp = pd.read_csv(os.path.join(RESULTS, "test_changepoints.csv"), parse_dates=["changepoint_date"])
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


def main():
    changepoints = load_test_changepoints()
    print(f"fine-grained test changepoints (n={len(changepoints)}): {[str(c.date()) for c in changepoints]}")
    naive_scale = naive_seasonal_mae_scale()

    all_rows = []
    for model_name, (info_level, variant, arch) in MODEL_FILES.items():
        raw = pd.read_parquet(os.path.join(EVAL_DIR, f"{model_name}_raw.parquet"))
        raw["regime"] = label_regime(raw["target_time"], changepoints)

        groups = [("overall", "overall", raw)]
        for hb in ["short_1-24h", "medium_48-168h"]:
            for rg in ["stable", "transition"]:
                sub = raw[(raw["horizon_bucket"] == hb) & (raw["regime"] == rg)]
                if len(sub) > 0:
                    groups.append((hb, rg, sub))

        for hb, rg, sub in groups:
            all_rows.append({
                "model": model_name, "architecture": arch, "variant": variant,
                "horizon_bucket": hb, "regime": rg, "n": len(sub),
                "MAE": sub["abs_err"].mean(),
                "RMSE": np.sqrt(sub["sq_err"].mean()),
                "MASE": sub["abs_err"].mean() / naive_scale,
                "pinball_avg": sub["pinball"].mean(),
                "CRPS_approx": sub["crps_approx"].mean(),
            })

    summary = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS, "step3_finegrained_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")

    pivot = summary[summary["horizon_bucket"] != "overall"].pivot_table(
        index=["architecture", "variant"], columns=["horizon_bucket", "regime"], values="MASE"
    )
    print("\n=== fine-grained MASE by horizon_bucket x regime ===")
    print(pivot.to_string())
    pivot.to_csv(os.path.join(RESULTS, "step3_mase_pivot.csv"))

    # covariate改善率(fine-grained版)
    archs = ["LightGBM", "LSTM", "TFT-lite"]
    rows = []
    for a in archs:
        for hb in ["short_1-24h", "medium_48-168h"]:
            for rg in ["stable", "transition"]:
                p = summary[(summary["architecture"] == a) & (summary["variant"] == "②power-only") &
                            (summary["horizon_bucket"] == hb) & (summary["regime"] == rg)]["MASE"].values
                c = summary[(summary["architecture"] == a) & (summary["variant"] == "③+covariate") &
                            (summary["horizon_bucket"] == hb) & (summary["regime"] == rg)]["MASE"].values
                if len(p) and len(c):
                    rows.append({
                        "architecture": a, "horizon_bucket": hb, "regime": rg,
                        "n_stable_or_transition": summary[(summary["architecture"] == a) & (summary["variant"] == "②power-only") &
                                                            (summary["horizon_bucket"] == hb) & (summary["regime"] == rg)]["n"].values[0],
                        "MASE_power_only": p[0], "MASE_with_covariate": c[0],
                        "improvement_pct": (p[0] - c[0]) / p[0] * 100,
                    })
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(RESULTS, "step3_covariate_improvement.csv"), index=False)
    print("\n=== fine-grained covariate improvement (%) ===")
    print(imp.to_string(index=False))
    return summary, imp


if __name__ == "__main__":
    main()
