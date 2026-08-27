"""
Step 4: TFT-lite③(power+covariate)のVariable Selection Network(VSN)重み分析

model_tft.pyのevaluate_on_testで保存済みの results/eval/tft_info3_variable_weights.csv
(origin単位, 直近168hのVSN重みを時間平均したもの)を、originの時刻がtest期間内の
安定期/遷移期どちらに属するかでラベル付けし、比較する。LightGBMのSHAP分析(shap_analysis.py)
と対になる「もう一つのモデルでも同じ問いに答える」分析。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reevaluate_finegrained import load_test_changepoints, label_regime

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval")

COV_FEATS = ["job_arrival_count_lag0h", "avg_requested_nodes_lag0h", "avg_requested_cores_lag0h",
             "new_user_count_lag0h", "avg_wait_time_sec_lag0h", "compute_bound_ratio_lag0h",
             "failure_rate_lag0h"]


def main():
    vw = pd.read_csv(os.path.join(EVAL_DIR, "tft_info3_variable_weights.csv"), parse_dates=["origin"])
    changepoints = load_test_changepoints()
    vw["regime"] = label_regime(vw["origin"], changepoints)

    feat_cols = [c for c in vw.columns if c not in ("origin", "regime")]

    overall = vw[feat_cols].mean().sort_values(ascending=False)
    print("=== mean VSN weight overall ===")
    print(overall)

    by_regime = vw.groupby("regime")[feat_cols].mean().T
    by_regime["transition_vs_stable_ratio"] = by_regime["transition"] / by_regime["stable"]
    by_regime = by_regime.sort_values("transition", ascending=False)
    by_regime.to_csv(os.path.join(RESULTS, "step4_tft_vsn_by_regime.csv"))
    print("\n=== VSN weight by regime (sorted by transition weight) ===")
    print(by_regime.to_string())

    cov_share = vw.assign(cov_weight=vw[COV_FEATS].sum(axis=1)).groupby("regime")["cov_weight"].mean()
    print("\n=== total covariate VSN weight share by regime ===")
    print(cov_share)
    cov_share.to_csv(os.path.join(RESULTS, "step4_tft_vsn_covariate_share.csv"))

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top = overall.head(10).iloc[::-1]
    colors = ["#f58518" if f in COV_FEATS else "#4c78a8" for f in top.index]
    axes[0].barh(top.index, top.values, color=colors)
    axes[0].set_title("TFT-lite3: top10 VSN weight (overall, time-avg over last 168h)")
    axes[0].set_xlabel("mean VSN weight")

    cov_only = by_regime.loc[[f for f in by_regime.index if f in COV_FEATS], ["stable", "transition"]]
    x = np.arange(len(cov_only))
    axes[1].bar(x - 0.2, cov_only["stable"], 0.4, label="stable", color="#4c78a8")
    axes[1].bar(x + 0.2, cov_only["transition"], 0.4, label="transition", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cov_only.index, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("mean VSN weight")
    axes[1].set_title("TFT-lite3: covariate VSN weight, stable vs transition")
    axes[1].legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS, "step4_tft_vsn.png")
    plt.savefig(fig_path, dpi=110)
    print(f"\nsaved: {fig_path}")


if __name__ == "__main__":
    main()
