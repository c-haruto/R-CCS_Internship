"""
変化点検出: 日次集計した消費電力・稼働ノード数・投入率・compute_bound比率の
多変量系列に対して ruptures (PELT, rbfコスト) を適用し、レジーム遷移点候補を求める。

train/valid/testの分割設計（step 3で使う「安定期」「遷移期」の定義にも使用）の根拠として、
results/changepoints.csv に保存する。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")

df = pd.read_parquet(os.path.join(RESULTS, "hourly_timeseries.parquet")).set_index("timestamp")

daily = df.resample("1D").agg({
    "power_consumption": "sum",
    "avg_active_nodes": "mean",
    "job_arrival_count": "sum",
    "compute_bound_ratio": "mean",
    "failure_rate": "mean",
    "avg_wait_time_sec": "mean",
})

# 立ち上げ期（共用開始2021-03-09前後）と末尾の不完全月は変化点検出から除外
RAMP_END = pd.Timestamp("2021-04-01", tz="Asia/Tokyo")
TAIL_START = pd.Timestamp("2024-04-25", tz="Asia/Tokyo")
analysis = daily[(daily.index >= RAMP_END) & (daily.index < TAIL_START)].copy()
analysis = analysis.dropna()

feature_cols = ["power_consumption", "avg_active_nodes", "job_arrival_count", "compute_bound_ratio"]
X = analysis[feature_cols].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)  # z-score normalize each series before joint detection

# PELT + rbf: 平均・分散どちらのシフトも拾える汎用コスト。penaltyは系列長に応じて調整。
algo = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(X)  # min_size=14日: 2週間未満の変化は無視
bkps = algo.predict(pen=15)
bkps_dates = [analysis.index[i - 1] for i in bkps[:-1]]  # 最後の要素は系列長そのものなので除外

print(f"multivariate changepoints (n={len(bkps_dates)}):")
for d in bkps_dates:
    print(" ", d.date())

# 主ターゲット(power_consumption)単独でも検出し、多変量版と突き合わせる
algo_p = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(analysis[["power_consumption"]].to_numpy())
bkps_p = algo_p.predict(pen=15)
bkps_p_dates = [analysis.index[i - 1] for i in bkps_p[:-1]]
print(f"\npower_consumption-only changepoints (n={len(bkps_p_dates)}):")
for d in bkps_p_dates:
    print(" ", d.date())

out = pd.DataFrame({
    "changepoint_date": bkps_dates,
    "source": "multivariate_rbf",
})
out2 = pd.DataFrame({
    "changepoint_date": bkps_p_dates,
    "source": "power_only_rbf",
})
all_cp = pd.concat([out, out2], ignore_index=True).sort_values("changepoint_date")
all_cp.to_csv(os.path.join(RESULTS, "changepoints.csv"), index=False)
print(f"\nsaved: {os.path.join(RESULTS, 'changepoints.csv')}")

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
axes[0].plot(daily.index, daily["power_consumption"], color="crimson", lw=0.8)
axes[0].set_title("Daily power_consumption with detected changepoints (multivariate, red) / (power-only, blue)")
for d in bkps_dates:
    axes[0].axvline(d, color="black", lw=1.2, alpha=0.7)
for d in bkps_p_dates:
    axes[0].axvline(d, color="blue", lw=0.8, ls="--", alpha=0.5)

axes[1].plot(daily.index, daily["avg_active_nodes"], color="purple", lw=0.8)
axes[1].set_title("Daily avg_active_nodes with same changepoints")
for d in bkps_dates:
    axes[1].axvline(d, color="black", lw=1.2, alpha=0.7)

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "changepoints_overview.png"), dpi=110)
print(f"saved: {os.path.join(RESULTS, 'changepoints_overview.png')}")
