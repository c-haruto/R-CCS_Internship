"""
Step 2: モデル②-TFT(電力のみ) / ③-TFT(電力+covariate)

Temporal Fusion Transformer簡易実装(TFT-lite)。外部ライブラリ(pytorch-forecasting等)への
依存リスクを避けるため、TFTの核となる2要素を手実装する:
  1. Variable Selection Network (VSN): 各入力特徴量チャンネルをGRN(Gated Residual Network)で
     埋め込み、softmaxゲートで重み付き合成する。時刻ごとの特徴量重要度(weights)がそのまま
     step4のcovariate重要度分析に使える。
  2. Interpretable Multi-Head Self-Attention: LSTMエンコーダ出力の直近168時間(週内)に対して
     self-attentionを適用する（720時間全体への attention はCPU計算コスト上、直近168hに限定）。
     attention重みも step4 で使える。

②③は同一アーキテクチャ・同一ハイパラ(README 2.1)。データセット/学習ループの土台は
model_lstm.pyのOriginDataset等を再利用し、モデル本体のみ差し替える。
test期間中は重み凍結(README 2.3)。

実行: OMP_NUM_THREADS=8 ~/dl_env/bin/python -u scripts/model_tft.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_table, load_split_ranges, MAX_LOOKBACK
from evaluation import evaluate, naive_seasonal_mae_scale, QUANTILES
from model_lstm import (
    OriginDataset, collate, seq_feature_cols, fit_scalers, pinball_loss_torch, HORIZONS_ARR,
)

torch.manual_seed(42)
torch.set_num_threads(8)

ATTN_WINDOW = 168  # self-attentionは直近168時間(週内)に限定してCPUコストを抑える


class GRN(nn.Module):
    """Gated Residual Network (TFT論文の基本ブロック)。"""

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        output_size = output_size or input_size
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        return self.ln(self.skip(x) + out * g)


class VariableSelectionNetwork(nn.Module):
    """特徴量チャンネルごとにGRNで埋め込み、softmaxゲートで重み付き合成する。"""

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.single_grns = nn.ModuleList([GRN(1, hidden_size, hidden_size) for _ in range(n_features)])
        self.weight_grn = GRN(n_features * hidden_size, hidden_size, n_features)

    def forward(self, x):  # x: [B,T,F]
        B, T, Fn = x.shape
        embeds = [self.single_grns[i](x[..., i:i + 1]) for i in range(Fn)]
        stacked = torch.stack(embeds, dim=-2)  # [B,T,F,hidden]
        flat = stacked.reshape(B, T, Fn * self.hidden_size)
        weights = F.softmax(self.weight_grn(flat), dim=-1)  # [B,T,F]
        combined = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # [B,T,hidden]
        return combined, weights


class TFTLite(nn.Module):
    def __init__(self, n_features, hidden=64, n_quantiles=5, cal_dim=7, n_heads=4):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=0.1)
        self.post_attn_grn = GRN(hidden, hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + 1 + cal_dim, 64), nn.ReLU(),
            nn.Linear(64, n_quantiles),
        )
        self.last_var_weights = None
        self.last_attn_weights = None

    def forward(self, seq, horizons, target_cal):
        combined, var_weights = self.vsn(seq)  # [B,T,hidden]
        lstm_out, (h_n, _) = self.lstm(combined)  # [B,T,hidden]

        recent = lstm_out[:, -ATTN_WINDOW:, :]
        attn_out, attn_weights = self.attn(recent, recent, recent, need_weights=True, average_attn_weights=True)
        fused = self.post_attn_grn(attn_out)  # [B,ATTN_WINDOW,hidden]
        pooled = fused[:, -1, :]  # 直近時刻の表現(=origin時点)をプールとして使用

        self.last_var_weights = var_weights.detach()
        self.last_attn_weights = attn_weights.detach()

        B, H = horizons.shape
        pooled_exp = pooled.unsqueeze(1).expand(-1, H, -1)
        x = torch.cat([pooled_exp, horizons.unsqueeze(-1), target_cal], dim=-1)
        return self.head(x)


def train_model(info_level, table, mean, std, epochs=15, batch_size=32, lr=1e-3):
    train_ds = OriginDataset(table, "train", info_level, origin_step=6, mean=mean, std=std)
    valid_ds = OriginDataset(table, "valid", info_level, origin_step=6, mean=mean, std=std)
    print(f"[TFT info_level={info_level}] train origins={len(train_ds)} valid origins={len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    n_features = len(seq_feature_cols(info_level))
    model = TFTLite(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience, bad_epochs = 3, 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss, n_batches = 0.0, 0
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
        val_loss, n_vbatches = 0.0, 0
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
    var_weight_records = []  # (origin, feature_idx -> mean weight over recent window) for step4

    pmean, pstd = mean["power_consumption"], std["power_consumption"]
    feat_cols = seq_feature_cols(info_level)
    with torch.no_grad():
        for seq, hs, cal, y, origins in loader:
            preds = model(seq, hs, cal).numpy()
            y_np = y.numpy()

            vw = model.last_var_weights.mean(dim=1).numpy()  # [B, n_features] time-averaged
            for bi, o in enumerate(origins):
                var_weight_records.append([o] + vw[bi].tolist())

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

    var_df = pd.DataFrame(var_weight_records, columns=["origin"] + feat_cols)
    return meta, y_true, quantile_preds, var_df


def run(info_level):
    t0 = time.time()
    table = build_feature_table()
    ranges = load_split_ranges()
    cols = seq_feature_cols(info_level)
    mean, std = fit_scalers(table, ranges["train"], cols)

    model = train_model(info_level, table, mean, std)
    meta, y_true, quantile_preds, var_df = evaluate_on_test(model, info_level, table, mean, std)

    scale = naive_seasonal_mae_scale()
    summary, raw = evaluate(f"tft_info{info_level}", info_level, meta, y_true, quantile_preds, naive_scale=scale)

    RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
    var_df.to_csv(os.path.join(RESULTS, "eval", f"tft_info{info_level}_variable_weights.csv"), index=False)

    print(f"\n[info_level={info_level}] TFT-lite done in {time.time()-t0:.1f}s")
    print(summary.to_string(index=False))
    print("\nmean variable-selection weight (test, time-averaged):")
    print(var_df.drop(columns=["origin"]).mean().sort_values(ascending=False))
    return summary


if __name__ == "__main__":
    all_summaries = []
    for info_level in ["2", "3"]:
        all_summaries.append(run(info_level))
    combined = pd.concat(all_summaries, ignore_index=True)
    RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
    combined.to_csv(os.path.join(RESULTS, "eval", "tft_combined_summary.csv"), index=False)
    print("\n=== combined ②③ TFT-lite summary ===")
    print(combined.to_string(index=False))
