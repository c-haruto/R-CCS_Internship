"""
Step 2: モデル②-LSTM(電力のみ) / ③-LSTM(電力+covariate)

LightGBMの手作りラグ/rolling特徴量とは対照的に、直近720時間(30日, README2.3のコンテキスト上限)の
生の時系列をLSTMエンコーダに読ませ、その最終隠れ状態 + horizon + 予測対象時刻のカレンダー特徴量を
MLPヘッドに通して分位点([0.1,0.25,0.5,0.75,0.9])を出力する(direct multi-horizon, 全HORIZONSを
1回のエンコードで共有するマルチタスク学習)。

②③は同一アーキテクチャ・同一ハイパラ(README 2.1): 入力系列の次元だけが異なる
(②は電力+カレンダー、③はさらにcovariate系列を追加)。
train+validで学習後、test期間中は重み凍結(README 2.3のフェアネス規約)。

実行はdl_env (torch導入済み) で行う: ~/dl_env/bin/python scripts/model_lstm.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_table, load_split_ranges, MAX_LOOKBACK, HORIZONS, COVARIATE_COLS
from evaluation import evaluate, naive_seasonal_mae_scale, QUANTILES

torch.manual_seed(42)
torch.set_num_threads(8)  # LSTMは逐次処理が支配的で、過剰スレッド化はオーバーヘッドを増やすだけのため制限

CAL_COLS = ["cal_hour_sin", "cal_hour_cos", "cal_dow_sin", "cal_dow_cos",
            "cal_doy_sin", "cal_doy_cos", "cal_is_weekend"]
HORIZONS_ARR = np.array(HORIZONS)
MAX_H = max(HORIZONS)


def seq_feature_cols(info_level):
    cols = ["power_consumption"] + CAL_COLS
    if info_level == "3":
        cols += [f"{c}_lag0h" for c in COVARIATE_COLS]
    return cols


def fit_scalers(table, train_range, cols):
    start, end = train_range
    sub = table.loc[(table.index >= start) & (table.index < end), cols]
    mean = sub.mean()
    std = sub.std().replace(0, 1.0)
    return mean, std


class OriginDataset(Dataset):
    def __init__(self, table, split_name, info_level, origin_step, mean, std):
        ranges = load_split_ranges()
        start, end = ranges[split_name]
        self.table = table
        self.cols = seq_feature_cols(info_level)
        self.mean, self.std = mean, std

        candidates = table.index[(table.index >= start) & (table.index < end)][::origin_step]
        earliest = table.index[0] + pd.Timedelta(hours=MAX_LOOKBACK - 1)
        origins = []
        for o in candidates:
            if o < earliest:
                continue
            if o + pd.Timedelta(hours=MAX_H) >= end:
                continue
            origins.append(o)
        self.origins = origins

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        o = self.origins[idx]
        window = self.table.loc[o - pd.Timedelta(hours=MAX_LOOKBACK - 1): o, self.cols]
        seq = ((window - self.mean) / self.std).to_numpy(dtype=np.float32)

        target_times = pd.DatetimeIndex([o + pd.Timedelta(hours=int(h)) for h in HORIZONS_ARR])
        y_raw = self.table.loc[target_times, "power_consumption"].to_numpy(dtype=np.float32)
        y = (y_raw - self.mean["power_consumption"]) / self.std["power_consumption"]

        hour = target_times.hour.to_numpy()
        dow = target_times.dayofweek.to_numpy()
        doy = target_times.dayofyear.to_numpy()
        target_cal = np.stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
            (dow >= 5).astype(np.float64),
        ], axis=1).astype(np.float32)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(HORIZONS_ARR.astype(np.float32) / MAX_H),
            torch.from_numpy(target_cal),
            torch.from_numpy(y),
            o,
        )


def collate(batch):
    seqs, hs, cals, ys, origins = zip(*batch)
    return (torch.stack(seqs), torch.stack(hs), torch.stack(cals), torch.stack(ys), list(origins))


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, n_quantiles=5, cal_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + 1 + cal_dim, 64), nn.ReLU(),
            nn.Linear(64, n_quantiles),
        )

    def forward(self, seq, horizons, target_cal):
        _, (h_n, _) = self.lstm(seq)
        pooled = h_n[-1]  # [B, hidden]
        B, H = horizons.shape
        pooled_exp = pooled.unsqueeze(1).expand(-1, H, -1)
        x = torch.cat([pooled_exp, horizons.unsqueeze(-1), target_cal], dim=-1)
        return self.head(x)  # [B, H, n_quantiles]


def pinball_loss_torch(y_true, preds, quantiles):
    # y_true: [B,H], preds: [B,H,Q]
    y = y_true.unsqueeze(-1)
    diff = y - preds
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device).view(1, 1, -1)
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()


def train_model(info_level, table, mean, std, epochs=15, batch_size=32, lr=1e-3):
    train_ds = OriginDataset(table, "train", info_level, origin_step=6, mean=mean, std=std)
    valid_ds = OriginDataset(table, "valid", info_level, origin_step=6, mean=mean, std=std)
    print(f"[info_level={info_level}] train origins={len(train_ds)} valid origins={len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    n_features = len(seq_feature_cols(info_level))
    model = LSTMForecaster(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience, bad_epochs = 3, 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0
        for seq, hs, cal, y, _ in train_loader:
            opt.zero_grad()
            preds = model(seq, hs, cal)
            loss = pinball_loss_torch(y, preds, QUANTILES)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        n_vbatches = 0
        with torch.no_grad():
            for seq, hs, cal, y, _ in valid_loader:
                preds = model(seq, hs, cal)
                val_loss += pinball_loss_torch(y, preds, QUANTILES).item()
                n_vbatches += 1
        val_loss /= n_vbatches

        print(f"  epoch {epoch+1}/{epochs} train_pinball={train_loss:.4f} valid_pinball={val_loss:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_on_test(model, info_level, table, mean, std):
    test_ds = OriginDataset(table, "test", info_level, origin_step=24, mean=mean, std=std)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    rows = {"origin": [], "horizon": [], "target_time": []}
    preds_by_q = {q: [] for q in QUANTILES}
    y_true_all = []

    pmean, pstd = mean["power_consumption"], std["power_consumption"]
    with torch.no_grad():
        for seq, hs, cal, y, origins in loader:
            preds = model(seq, hs, cal).numpy()  # [B,H,Q] standardized scale
            y_np = y.numpy()
            B, H, Q = preds.shape
            for bi in range(B):
                o = origins[bi]
                for hi, h in enumerate(HORIZONS_ARR):
                    rows["origin"].append(o)
                    rows["horizon"].append(int(h))
                    rows["target_time"].append(o + pd.Timedelta(hours=int(h)))
                    y_true_all.append(y_np[bi, hi] * pstd + pmean)
                    for qi, q in enumerate(QUANTILES):
                        preds_by_q[q].append(preds[bi, hi, qi] * pstd + pmean)

    meta = pd.DataFrame(rows)
    y_true = np.array(y_true_all)
    quantile_preds = {q: np.array(v) for q, v in preds_by_q.items()}
    return meta, y_true, quantile_preds


def run(info_level):
    t0 = time.time()
    table = build_feature_table()
    ranges = load_split_ranges()
    cols = seq_feature_cols(info_level)
    mean, std = fit_scalers(table, ranges["train"], cols)

    model = train_model(info_level, table, mean, std)
    meta, y_true, quantile_preds = evaluate_on_test(model, info_level, table, mean, std)

    scale = naive_seasonal_mae_scale()
    summary, raw = evaluate(f"lstm_info{info_level}", info_level, meta, y_true, quantile_preds, naive_scale=scale)

    print(f"\n[info_level={info_level}] LSTM done in {time.time()-t0:.1f}s")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    all_summaries = []
    for info_level in ["2", "3"]:
        all_summaries.append(run(info_level))
    combined = pd.concat(all_summaries, ignore_index=True)
    RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
    combined.to_csv(os.path.join(RESULTS, "eval", "lstm_combined_summary.csv"), index=False)
    print("\n=== combined ②③ LSTM summary ===")
    print(combined.to_string(index=False))
