"""
Step 4: LightGBM③(power+covariate)のSHAP分析

- 中央値(q=0.5)モデルを再学習(model_lightgbm.pyと同一条件、決定論的なので同一モデルになる)し、
  test setに対するSHAP値(shap.TreeExplainer, 厳密解)を計算する。
- どの特徴量(特にcovariate由来)が予測に効いているかを、全体・horizon帯・レジーム(安定期/遷移期,
  results/test_changepoints.csvベース)別に比較する。
- 「covariateはどれだけ寄与しているか」「その寄与は遷移期で高まるか」を定量化し、
  運用上監視すべき早期シグナルの示唆を得る(README Step4参照)。
"""
import os

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import build_feature_table, make_direct_horizon_samples, covariate_columns, power_only_columns
from model_lightgbm import LGB_PARAMS
from reevaluate_finegrained import load_test_changepoints, label_regime

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")


def main():
    table = build_feature_table()
    cov_cols = set(covariate_columns(table))
    power_cols = set(power_only_columns(table))

    X_train, y_train, _ = make_direct_horizon_samples(table, "train", "3", origin_step=6)
    X_valid, y_valid, _ = make_direct_horizon_samples(table, "valid", "3", origin_step=6)
    X_test, y_test, meta_test = make_direct_horizon_samples(table, "test", "3", origin_step=24)
    print(f"train={X_train.shape} valid={X_valid.shape} test={X_test.shape}")

    model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LGB_PARAMS)
    model.fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="quantile",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    print("best_iteration:", model.best_iteration_)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("shap_values shape:", shap_values.shape)

    feature_names = X_test.columns.to_numpy()
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    importance.to_csv(os.path.join(RESULTS, "step4_shap_overall_importance.csv"), header=["mean_abs_shap"])
    print("\ntop 20 features (overall mean |SHAP|):")
    print(importance.head(20))

    def category(f):
        if f in cov_cols:
            return "covariate"
        if f in power_cols:
            return "power/calendar"
        if f.startswith("target_"):
            return "target_time_calendar"
        return "other"

    cat = pd.Series([category(f) for f in feature_names], index=feature_names)

    changepoints = load_test_changepoints()
    regime = label_regime(meta_test["target_time"], changepoints)
    hbucket = np.where(meta_test["horizon"].to_numpy() <= 24, "short_1-24h", "medium_48-168h")

    rows = []
    for hb in ["overall", "short_1-24h", "medium_48-168h"]:
        for rg in ["overall", "stable", "transition"]:
            mask = np.ones(len(X_test), dtype=bool)
            if hb != "overall":
                mask &= (hbucket == hb)
            if rg != "overall":
                mask &= (regime == rg)
            if mask.sum() == 0:
                continue
            sub_shap = np.abs(shap_values[mask])
            total = sub_shap.sum()
            cat_share = {}
            for c in ["covariate", "power/calendar", "target_time_calendar"]:
                idx = np.where(cat.to_numpy() == c)[0]
                cat_share[c] = sub_shap[:, idx].sum() / total * 100
            rows.append({"horizon_bucket": hb, "regime": rg, "n": int(mask.sum()), **cat_share})

    cat_df = pd.DataFrame(rows)
    cat_df.to_csv(os.path.join(RESULTS, "step4_shap_category_share.csv"), index=False)
    print("\n=== SHAP magnitude share (%) by category, horizon_bucket x regime ===")
    print(cat_df.to_string(index=False))

    # covariate特徴量だけのランキング (stable vs transition)
    cov_idx = np.where(cat.to_numpy() == "covariate")[0]
    cov_rows = []
    for rg in ["stable", "transition"]:
        mask = regime == rg
        m = np.abs(shap_values[mask][:, cov_idx]).mean(axis=0)
        for f, v in zip(feature_names[cov_idx], m):
            cov_rows.append({"feature": f, "regime": rg, "mean_abs_shap": v})
    cov_df = pd.DataFrame(cov_rows)
    cov_pivot = cov_df.pivot(index="feature", columns="regime", values="mean_abs_shap")
    cov_pivot["transition_vs_stable_ratio"] = cov_pivot["transition"] / cov_pivot["stable"]
    cov_pivot = cov_pivot.sort_values("transition", ascending=False)
    cov_pivot.to_csv(os.path.join(RESULTS, "step4_shap_covariate_by_regime.csv"))
    print("\n=== covariate features: mean |SHAP| stable vs transition (top 15 by transition importance) ===")
    print(cov_pivot.head(15).to_string())

    # --- plots ---
    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = importance.head(20).iloc[::-1]
    colors = ["#f58518" if f in cov_cols else "#4c78a8" for f in top20.index]
    ax.barh(top20.index, top20.values, color=colors)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("LightGBM3 (power+covariate): top20 feature importance (SHAP)\norange=covariate, blue=power/calendar")
    plt.tight_layout()
    fig1 = os.path.join(RESULTS, "step4_shap_top20.png")
    plt.savefig(fig1, dpi=110)
    plt.close()
    print(f"\nsaved: {fig1}")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = cat_df[cat_df["regime"].isin(["stable", "transition"]) & (cat_df["horizon_bucket"] == "overall")]
    x = np.arange(len(plot_df))
    ax.bar(x - 0.2, plot_df["covariate"], 0.2, label="covariate", color="#f58518")
    ax.bar(x, plot_df["power/calendar"], 0.2, label="power/calendar", color="#4c78a8")
    ax.bar(x + 0.2, plot_df["target_time_calendar"], 0.2, label="target_time_calendar", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["regime"])
    ax.set_ylabel("share of total |SHAP| (%)")
    ax.set_title("LightGBM3: SHAP magnitude share by feature category, stable vs transition")
    ax.legend()
    plt.tight_layout()
    fig2 = os.path.join(RESULTS, "step4_shap_category_share.png")
    plt.savefig(fig2, dpi=110)
    plt.close()
    print(f"saved: {fig2}")


if __name__ == "__main__":
    main()
