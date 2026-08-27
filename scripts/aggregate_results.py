"""
Step 2: 全7モデル(①Chronos + ②③×{LightGBM,LSTM,TFT-lite})の結果を集約し、
比較表・比較図を作成する。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval")

ARCH_MAP = {
    "chronos_bolt_small": "Chronos-bolt (zero-shot)",
    "lightgbm_info2": "LightGBM", "lightgbm_info3": "LightGBM",
    "lstm_info2": "LSTM", "lstm_info3": "LSTM",
    "tft_info2": "TFT-lite", "tft_info3": "TFT-lite",
}
COV_MAP = {
    "chronos_bolt_small": "①zero-shot",
    "lightgbm_info2": "②power-only", "lightgbm_info3": "③+covariate",
    "lstm_info2": "②power-only", "lstm_info3": "③+covariate",
    "tft_info2": "②power-only", "tft_info3": "③+covariate",
}


def load_all():
    files = [
        "chronos_bolt_small_summary.csv",
        "lightgbm_info2_summary.csv", "lightgbm_info3_summary.csv",
        "lstm_info2_summary.csv", "lstm_info3_summary.csv",
        "tft_info2_summary.csv", "tft_info3_summary.csv",
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(EVAL_DIR, f))
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["architecture"] = all_df["model"].map(ARCH_MAP)
    all_df["variant"] = all_df["model"].map(COV_MAP)
    return all_df


def main():
    all_df = load_all()
    out_csv = os.path.join(RESULTS, "step2_all_models_summary.csv")
    all_df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")

    overall = all_df[all_df["horizon_bucket"] == "overall"][
        ["architecture", "variant", "MAE", "RMSE", "MASE", "pinball_avg", "CRPS_approx"]
    ].sort_values("MASE")
    print("\n=== overall (test全体) ===")
    print(overall.to_string(index=False))

    # --- ①②③効果を軸ごとに要約するピボット (MASE, horizon_bucket×regime別) ---
    detail = all_df[all_df["horizon_bucket"] != "overall"]
    pivot = detail.pivot_table(index=["architecture", "variant"], columns=["horizon_bucket", "regime"], values="MASE")
    pivot_path = os.path.join(RESULTS, "step2_mase_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"\nsaved: {pivot_path}")
    print("\n=== MASE by horizon_bucket x regime ===")
    print(pivot.to_string())

    # --- 図1: overall MASE 横棒グラフ (全7モデル) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{a} {v}" for a, v in zip(overall["architecture"], overall["variant"])]
    colors = ["#999999" if "zero-shot" in v else ("#4c78a8" if "power-only" in v else "#f58518")
              for v in overall["variant"]]
    ax.barh(labels[::-1], overall["MASE"][::-1], color=colors[::-1])
    ax.set_xlabel("MASE (test overall, lower is better)")
    ax.set_title("Step2: 7-model comparison (test overall)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig1_path = os.path.join(RESULTS, "step2_overall_mase.png")
    plt.savefig(fig1_path, dpi=110)
    plt.close()
    print(f"saved: {fig1_path}")

    # --- 図2: ②vs③ MASE (アーキテクチャ×horizon_bucket×regime) covariate効果 ---
    archs = ["LightGBM", "LSTM", "TFT-lite"]
    hbs = ["short_1-24h", "medium_48-168h"]
    regimes = ["stable", "transition"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    width = 0.35
    for ax, hb in zip(axes, hbs):
        x = np.arange(len(archs) * len(regimes))
        xticklabels = []
        power_vals, cov_vals = [], []
        for a in archs:
            for r in regimes:
                p = detail[(detail["architecture"] == a) & (detail["variant"] == "②power-only") &
                           (detail["horizon_bucket"] == hb) & (detail["regime"] == r)]["MASE"]
                c = detail[(detail["architecture"] == a) & (detail["variant"] == "③+covariate") &
                           (detail["horizon_bucket"] == hb) & (detail["regime"] == r)]["MASE"]
                power_vals.append(p.values[0] if len(p) else np.nan)
                cov_vals.append(c.values[0] if len(c) else np.nan)
                xticklabels.append(f"{a}\n{r}")
        ax.bar(x - width / 2, power_vals, width, label="②power-only", color="#4c78a8")
        ax.bar(x + width / 2, cov_vals, width, label="③+covariate", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_title(f"horizon: {hb}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("MASE (lower is better)")
    axes[0].legend()
    plt.suptitle("Step2: covariate effect (②power-only vs ③+covariate) by architecture x regime")
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS, "step2_covariate_effect.png")
    plt.savefig(fig2_path, dpi=110)
    plt.close()
    print(f"saved: {fig2_path}")

    # --- covariate効果の相対改善率テーブル (③がstableよりtransitionでより効くか) ---
    rows = []
    for a in archs:
        for hb in hbs:
            for r in regimes:
                p = detail[(detail["architecture"] == a) & (detail["variant"] == "②power-only") &
                           (detail["horizon_bucket"] == hb) & (detail["regime"] == r)]["MASE"].values
                c = detail[(detail["architecture"] == a) & (detail["variant"] == "③+covariate") &
                           (detail["horizon_bucket"] == hb) & (detail["regime"] == r)]["MASE"].values
                if len(p) and len(c):
                    improvement_pct = (p[0] - c[0]) / p[0] * 100
                    rows.append({"architecture": a, "horizon_bucket": hb, "regime": r,
                                 "MASE_power_only": p[0], "MASE_with_covariate": c[0],
                                 "improvement_pct": improvement_pct})
    improvement_df = pd.DataFrame(rows)
    imp_path = os.path.join(RESULTS, "step2_covariate_improvement.csv")
    improvement_df.to_csv(imp_path, index=False)
    print(f"\nsaved: {imp_path}")
    print("\n=== covariate導入によるMASE改善率(%) (プラスが改善) ===")
    print(improvement_df.to_string(index=False))

    return all_df, overall, pivot, improvement_df


if __name__ == "__main__":
    main()
