import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
df = pd.read_parquet(os.path.join(RESULTS, "hourly_timeseries.parquet"))
df = df.set_index("timestamp")

daily = df.resample("1D").agg({
    "power_consumption": "sum",
    "avg_active_nodes": "mean",
    "job_arrival_count": "sum",
    "avg_concurrent_jobs": "mean",
    "failure_rate": "mean",
    "avg_wait_time_sec": "mean",
})

fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)

axes[0].plot(daily.index, daily["power_consumption"], color="crimson", lw=0.8)
axes[0].set_title("Daily total power_consumption (prorated econ, target variable)")

axes[1].plot(daily.index, daily["avg_active_nodes"], color="purple", lw=0.8)
axes[1].axhline(158976, color="gray", ls="--", lw=0.8, label="Fugaku total nodes (158976)")
axes[1].set_title("Daily mean avg_active_nodes")
axes[1].legend()

axes[2].plot(daily.index, daily["job_arrival_count"], color="darkgreen", lw=0.8)
axes[2].set_title("Daily job_arrival_count")

axes[3].plot(daily.index, daily["failure_rate"], color="darkorange", lw=0.8)
axes[3].set_title("Daily mean failure_rate")

axes[4].plot(daily.index, daily["avg_wait_time_sec"] / 3600, color="steelblue", lw=0.8)
axes[4].set_title("Daily mean wait time (hours)")

for ax in axes:
    ax.axvline(pd.Timestamp("2022-11-30", tz="Asia/Tokyo"), color="black", ls=":", lw=1,
               label="2022-11-30 (ChatGPT release / prior workload-shift finding)")
    ax.grid(True, alpha=0.3)
axes[0].legend()

plt.tight_layout()
out_path = os.path.join(RESULTS, "hourly_timeseries_overview.png")
plt.savefig(out_path, dpi=110)
print("saved", out_path)
