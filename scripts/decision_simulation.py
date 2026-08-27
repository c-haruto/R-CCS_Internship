"""
Step 5: 予測を電力・冷却容量の引当(provisioning)判断に接続する意思決定シミュレーション

枠組み(newsvendor / 非対称コスト最適化):
  各予測時点で「どれだけの冷却容量(=電力バジェット)を引き当てるか」を決める。
  引き当てた容量 C に対し実際の消費電力 y が上回れば「過小引当」(緊急冷却・スロットリング等の
  高コスト)、下回れば「過大引当」(遊休容量の無駄、低コスト)が発生する:
      cost(y, C) = r * max(y - C, 0) + 1 * max(C - y, 0)
  ここで r = 過小引当コスト / 過大引当コスト の比。この最小化問題の最適解は分位点予測そのもの
  (臨界分位点 q* = r/(1+r))であり、まさに①②③の学習・評価に使ったpinball lossと同じ数学。
  つまり「予測精度(pinball loss)の改善」が「引当コストの削減」に直結することを、
  具体的な運用コスト数値で示す。

  r=3 -> q*=0.75 (中程度のリスク回避: 過小引当は過大引当の3倍コスト)
  r=9 -> q*=0.90 (高リスク回避: ミッションクリティカルな冷却系を想定)
  どちらも既に計算済みの分位点(0.75, 0.90)にちょうど一致するため、追加学習は不要。

比較対象:
  - static: train期間の無条件分位点を定数として常に引き当てる(=予測を一切使わない安全マージン方式)
  - seasonal_naive: 24時間前の実測値 + train期間の残差分位点(単純な季節ナイーブ予測)
  - 7モデル(①Chronos, ②③×{LightGBM,LSTM,TFT-lite})の分位点予測

出力は正規化コスト単位(過大引当コスト=1/unit power)。econ列の物理単位(円やkWhとの換算)は
データセットに明記されていないため、絶対金額ではなく相対的なコスト削減率で実務的価値を示す。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import build_feature_table, load_split_ranges
from reevaluate_finegrained import load_test_changepoints, label_regime

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
EVAL_DIR = os.path.join(RESULTS, "eval")

MODEL_FILES = {
    "chronos_bolt_small": "①Chronos",
    "lightgbm_info2": "②LightGBM", "lightgbm_info3": "③LightGBM",
    "lstm_info2": "②LSTM", "lstm_info3": "③LSTM",
    "tft_info2": "②TFT-lite", "tft_info3": "③TFT-lite",
}
SCENARIOS = {"moderate (r=3)": (3, 0.75), "high-stakes (r=9)": (9, 0.90)}


def cost(y, C, r):
    return r * np.maximum(y - C, 0) + 1 * np.maximum(C - y, 0)


def main():
    table = build_feature_table()
    ranges = load_split_ranges()
    train_start, train_end = ranges["train"]
    train_power = table.loc[(table.index >= train_start) & (table.index < train_end), "power_consumption"]

    # --- shared meta/y_true (identical across all 7 models' raw files) ---
    base = pd.read_parquet(os.path.join(EVAL_DIR, "lightgbm_info2_raw.parquet"),
                            columns=["origin", "horizon", "target_time", "y_true", "horizon_bucket"])
    base["target_time"] = pd.to_datetime(base["target_time"])
    changepoints = load_test_changepoints()
    base["regime"] = label_regime(base["target_time"], changepoints)

    # --- static baseline: train期間の無条件分位点(定数) ---
    static_q = {q: train_power.quantile(q) for _, q in SCENARIOS.values()}

    # --- seasonal-naive baseline: y(target_time - 24h)の実測値 + train残差分位点 ---
    naive_point = table.reindex(base["target_time"] - pd.Timedelta(hours=24))["power_consumption"].to_numpy()
    train_resid = (train_power - train_power.shift(24)).dropna()
    naive_resid_q = {q: train_resid.quantile(q) for _, q in SCENARIOS.values()}

    policies = {"static": None, "seasonal_naive": None}
    quantile_preds_by_model = {}
    for model_name in MODEL_FILES:
        raw = pd.read_parquet(os.path.join(EVAL_DIR, f"{model_name}_raw.parquet"),
                               columns=["origin", "horizon", "target_time"] +
                                       [f"y_pred_q{q}" for _, q in set(SCENARIOS.values())])
        raw["target_time"] = pd.to_datetime(raw["target_time"])
        merged = base[["origin", "horizon", "target_time"]].merge(raw, on=["origin", "horizon", "target_time"], how="left")
        assert merged.shape[0] == base.shape[0], f"row mismatch for {model_name}"
        quantile_preds_by_model[model_name] = merged

    rows = []
    calib_rows = []
    for scen_name, (r, q) in SCENARIOS.items():
        y = base["y_true"].to_numpy()

        C_static = np.full(len(base), static_q[q])
        C_naive = naive_point + naive_resid_q[q]

        all_policies = {"static (no forecast)": C_static, "seasonal-naive": C_naive}
        for model_name, label in MODEL_FILES.items():
            all_policies[label] = quantile_preds_by_model[model_name][f"y_pred_q{q}"].to_numpy()

        for policy_name, C in all_policies.items():
            valid = ~np.isnan(C)
            per_row_cost = cost(y[valid], C[valid], r)
            coverage = float((y[valid] <= C[valid]).mean())  # 実現値が引当容量を超えなかった割合(≈qが望ましい)

            df = pd.DataFrame({
                "cost": per_row_cost,
                "horizon_bucket": base["horizon_bucket"].to_numpy()[valid],
                "regime": base["regime"].to_numpy()[valid],
            })
            for hb in ["overall", "short_1-24h", "medium_48-168h"]:
                for rg in ["overall", "stable", "transition"]:
                    sub = df
                    if hb != "overall":
                        sub = sub[sub["horizon_bucket"] == hb]
                    if rg != "overall":
                        sub = sub[sub["regime"] == rg]
                    rows.append({
                        "scenario": scen_name, "policy": policy_name, "q": q, "r": r,
                        "horizon_bucket": hb, "regime": rg,
                        "n": len(sub), "mean_cost": sub["cost"].mean(),
                    })
            calib_rows.append({"scenario": scen_name, "policy": policy_name, "q": q,
                                "target_coverage": q, "empirical_coverage": coverage})

    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(os.path.join(RESULTS, "step5_decision_cost.csv"), index=False)

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(os.path.join(RESULTS, "step5_calibration.csv"), index=False)

    print("=== calibration (empirical coverage vs target quantile) ===")
    print(calib_df.to_string(index=False))

    print("\n=== mean normalized cost, overall (horizon=overall, regime=overall) ===")
    overall = cost_df[(cost_df["horizon_bucket"] == "overall") & (cost_df["regime"] == "overall")]
    for scen in SCENARIOS:
        sub = overall[overall["scenario"] == scen].sort_values("mean_cost")
        print(f"\n--- {scen} ---")
        print(sub[["policy", "mean_cost"]].to_string(index=False))
        static_cost = sub[sub["policy"] == "static (no forecast)"]["mean_cost"].values[0]
        naive_cost = sub[sub["policy"] == "seasonal-naive"]["mean_cost"].values[0]
        best = sub[~sub["policy"].isin(["static (no forecast)", "seasonal-naive"])].iloc[0]
        print(f"best model: {best['policy']} ({best['mean_cost']:.3g})")
        print(f"  vs static:  {(1 - best['mean_cost']/static_cost)*100:+.1f}% cost")
        print(f"  vs naive:   {(1 - best['mean_cost']/naive_cost)*100:+.1f}% cost")

    print("\n=== mean normalized cost, by regime (short-horizon only, r=9 scenario) ===")
    by_regime = cost_df[(cost_df["horizon_bucket"] == "short_1-24h") & (cost_df["scenario"] == "high-stakes (r=9)") &
                         (cost_df["regime"].isin(["stable", "transition"]))]
    pivot = by_regime.pivot(index="policy", columns="regime", values="mean_cost")
    pivot.to_csv(os.path.join(RESULTS, "step5_cost_by_regime.csv"))
    print(pivot.sort_values("transition").to_string())

    # --- plot: cost by policy, r=9 scenario, overall ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, scen in zip(axes, SCENARIOS):
        sub = overall[overall["scenario"] == scen].sort_values("mean_cost")
        colors = []
        for p in sub["policy"]:
            if p == "static (no forecast)":
                colors.append("#78766f")
            elif p == "seasonal-naive":
                colors.append("#b3731f")
            elif p.startswith("①"):
                colors.append("#78766f")
            elif "LightGBM" in p:
                colors.append("#2a78d6")
            elif "LSTM" in p:
                colors.append("#eb6834")
            else:
                colors.append("#1baf7a")
        ax.barh(sub["policy"], sub["mean_cost"], color=colors)
        ax.set_title(f"{scen}")
        ax.set_xlabel("mean normalized provisioning cost")
        ax.invert_yaxis()
    plt.suptitle("Step5: capacity-provisioning cost by policy (lower is better)")
    plt.tight_layout()
    fig_path = os.path.join(RESULTS, "step5_decision_cost.png")
    plt.savefig(fig_path, dpi=110)
    print(f"\nsaved: {fig_path}")


if __name__ == "__main__":
    main()
