"""
Step 3: test期間(2022-10-01〜2024-04-25)内部により高感度な変化点検出を適用する。

split設計時(detect_changepoints.py)はtrain/valid/test境界を決めるための粗い検出
(pen=15, 2021-04〜2024-04全体で4点)だったため、test内部には2022-11-08しか
含まれておらず、step3の「安定期 vs 遷移期」比較のサンプル数(n=673/174)が少なすぎた。
ここではtest期間だけに絞り、より低いpenaltyで多変量+単変量の変化点を検出し、
安定期/遷移期のラベリングを精緻化する。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")

df = pd.read_parquet(os.path.join(RESULTS, "hourly_timeseries.parquet")).set_index("timestamp")
daily = df.resample("1D").agg({
    "power_consumption": "sum",
    "avg_active_nodes": "mean",
    "job_arrival_count": "sum",
    "avg_wait_time_sec": "mean",
    "compute_bound_ratio": "mean",
    "failure_rate": "mean",
})

with open(os.path.join(RESULTS, "split_design.json")) as f:
    import json
    split = json.load(f)
tz = daily.index.tz
test_start = pd.Timestamp(split["test"]["start"], tz=tz)
test_end = pd.Timestamp(split["test"]["end"], tz=tz)
test_daily = daily[(daily.index >= test_start) & (daily.index < test_end)].dropna()
print(f"test period: {test_daily.index.min().date()} .. {test_daily.index.max().date()} ({len(test_daily)} days)")

feature_cols = ["power_consumption", "avg_active_nodes", "job_arrival_count", "avg_wait_time_sec", "compute_bound_ratio"]
X = test_daily[feature_cols].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

results = {}
for pen in [4, 6, 8, 10]:
    algo = rpt.Pelt(model="rbf", min_size=10, jump=1).fit(X)
    bkps = algo.predict(pen=pen)
    dates = [test_daily.index[i - 1] for i in bkps[:-1]]
    results[pen] = dates
    print(f"pen={pen}: n={len(dates)}  {[str(d.date()) for d in dates]}")

# 単変量(power)でも参照用に検出
algo_p = rpt.Pelt(model="rbf", min_size=10, jump=1).fit(test_daily[["power_consumption"]].to_numpy())
bkps_p = algo_p.predict(pen=6)
dates_p = [test_daily.index[i - 1] for i in bkps_p[:-1]]
print(f"\npower-only pen=6: n={len(dates_p)}  {[str(d.date()) for d in dates_p]}")

# pen=6(多変量)を採用: 大まかな目安として1〜2ヶ月に1点程度の解像度になる penalty を選択
CHOSEN_PEN = 6
chosen_dates = results[CHOSEN_PEN]

out = pd.DataFrame({"changepoint_date": chosen_dates, "source": f"test_multivariate_rbf_pen{CHOSEN_PEN}"})
out_path = os.path.join(RESULTS, "test_changepoints.csv")
out.to_csv(out_path, index=False)
print(f"\nsaved: {out_path} (n={len(out)})")

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
axes[0].plot(test_daily.index, test_daily["power_consumption"], color="crimson", lw=0.8)
axes[0].set_title(f"test period daily power_consumption with fine-grained changepoints (pen={CHOSEN_PEN})")
axes[1].plot(test_daily.index, test_daily["job_arrival_count"], color="darkgreen", lw=0.8)
axes[1].set_title("job_arrival_count")
axes[2].plot(test_daily.index, test_daily["avg_wait_time_sec"] / 3600, color="steelblue", lw=0.8)
axes[2].set_title("avg_wait_time_sec (hours)")
for ax in axes:
    for d in chosen_dates:
        ax.axvline(d, color="black", lw=1, ls="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(RESULTS, "test_changepoints_overview.png")
plt.savefig(fig_path, dpi=110)
print(f"saved: {fig_path}")
