"""
発展分析(avg_active_nodes): results/test_changepoints.csv の4変化点でregimeを再ラベルし、
安定期/遷移期別のMASEを集計する。reevaluate_finegrained.pyのnodes版。
"""
import os

import numpy as np
import pandas as pd

from evaluation_nodes import naive_seasonal_mae_scale, TRANSITION_WINDOW_DAYS

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval_nodes")

MODEL_FILES = {
    "chronos_bolt_small_nodes": ("1", "①zero-shot", "Chronos-bolt (zero-shot)"),
    "lightgbm_info2_nodes": ("2", "②power-only", "LightGBM"),
    "lightgbm_info3_nodes": ("3", "③+covariate", "LightGBM"),
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
            })

    summary = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS, "step_ext_nodes_finegrained_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")

    pivot = summary[summary["horizon_bucket"] != "overall"].pivot_table(
        index=["architecture", "variant"], columns=["horizon_bucket", "regime"], values="MASE"
    )
    print("\n=== avg_active_nodes: fine-grained MASE by horizon_bucket x regime ===")
    print(pivot.to_string())
    pivot.to_csv(os.path.join(RESULTS, "step_ext_nodes_mase_pivot.csv"))
    return summary


if __name__ == "__main__":
    main()
