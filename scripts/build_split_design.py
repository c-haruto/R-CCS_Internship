"""
Step 2 準備: train/valid/test の時間分割を確定し、根拠(変化点)とともに
results/split_design.json / split_design_overview.png に保存する。

設計方針 v2（詳細はREADME参照。当初のv1分割から研究の主題を変更したため改訂）:
- 研究の主題を「①時系列LLM vs ②③訓練済みML」の全体比較に置き直し、
  「遷移期でこそ③が優位か」の検証は副次的な観察に格下げしたため、
  trainをできるだけ厚くしてMLモデルの学習データ不足という交絡要因を減らす方向に再設計。
- 立ち上げ期(2021-03)と末尾の不完全月(2024-04-25以降)は全splitから除外（v1と同じ）。
- TRAIN: 2021-04-01 - 2023-01-01 (約640日、v1の456日から+40%)  変化点4件全て
  (2021-06-15, 2022-04-10, 2022-07-23, 2022-11-08)を含む。
- VALID: 2023-01-01 - 2023-03-01 (約59日)  ハイパラ選択・early stoppingのみが目的で、
  v1のように変化点を含めることは要件としない。
- TEST : 2023-03-01 - 2024-04-25 (約421日)  train/validに含まれない期間。
  ここに対してさらに高感度な変化点検出を適用し、安定期/遷移期のサブウィンドウ分析を
  副次的に行う(step 3)。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")

df = pd.read_parquet(os.path.join(RESULTS, "hourly_timeseries.parquet")).set_index("timestamp")
tz = df.index.tz

SPLITS = {
    "ramp_up_excluded": ("2021-03-01", "2021-04-01"),
    "train": ("2021-04-01", "2023-01-01"),
    "valid": ("2023-01-01", "2023-03-01"),
    "test": ("2023-03-01", "2024-04-25"),
    "tail_excluded": ("2024-04-25", "2024-05-15"),
}

changepoints = pd.read_csv(os.path.join(RESULTS, "changepoints.csv"), parse_dates=["changepoint_date"])

summary = {}
for name, (start, end) in SPLITS.items():
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    mask = (df.index >= start_ts) & (df.index < end_ts)
    n_hours = int(mask.sum())
    cps_in = changepoints[
        (changepoints["changepoint_date"] >= start_ts) & (changepoints["changepoint_date"] < end_ts)
    ]["changepoint_date"].dt.date.astype(str).tolist()
    summary[name] = {
        "start": start, "end": end, "n_hours": n_hours,
        "n_days": round(n_hours / 24, 1),
        "changepoints_inside": sorted(set(cps_in)),
    }

print(json.dumps(summary, indent=2, ensure_ascii=False))

with open(os.path.join(RESULTS, "split_design.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nsaved: {os.path.join(RESULTS, 'split_design.json')}")

# --- plot ---
daily = df["power_consumption"].resample("1D").sum()
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily.index, daily.values, color="black", lw=0.7)

colors = {"ramp_up_excluded": "lightgray", "train": "#a6cee3", "valid": "#fdbf6f",
          "test": "#b2df8a", "tail_excluded": "lightgray"}
for name, (start, end) in SPLITS.items():
    ax.axvspan(pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz), color=colors[name], alpha=0.4, label=name)

for _, row in changepoints.iterrows():
    ax.axvline(row["changepoint_date"], color="red", lw=1, ls="--", alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")
ax.set_title("Train / Valid / Test split with detected changepoints (red dashed)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "split_design_overview.png"), dpi=110)
print(f"saved: {os.path.join(RESULTS, 'split_design_overview.png')}")
