"""
Step 2: モデル① Chronos-bolt zero-shot (電力のみ, 学習なし)

- amazon/chronos-bolt-small を使用、CPU推論。
- LightGBM(model_lightgbm.py)と全く同じtest起点集合・horizon集合・評価指標で比較する
  (features.make_direct_horizon_samples(table, 'test', info_level, origin_step=24) を再利用)。
- コンテキストは直近720時間(30日)に統一（README 2.3のフェアネス規約: MLモデルのラグ特徴量と
  参照可能な過去情報の長さを揃える）。
- 重み更新は一切行わない（zero-shotなので自動的にtest期間中凍結の規約を満たす）。

実行はdl_env (torch/transformers/chronos-forecasting導入済み) で行う:
  ~/dl_env/bin/python scripts/model_chronos.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_table, make_direct_horizon_samples, MAX_LOOKBACK, HORIZONS
from evaluation import evaluate, naive_seasonal_mae_scale, QUANTILES

MODEL_ID = "amazon/chronos-bolt-small"
MAX_HORIZON = max(HORIZONS)  # 168
BATCH_SIZE = 32


def build_contexts(table, origins):
    """originごとに直近MAX_LOOKBACK時間分のpower_consumptionをtorch.Tensorで返す。"""
    contexts = []
    for o in origins:
        window = table.loc[:o, "power_consumption"].iloc[-MAX_LOOKBACK:]
        assert len(window) == MAX_LOOKBACK, f"insufficient context at {o}: {len(window)}"
        contexts.append(torch.tensor(window.to_numpy(), dtype=torch.float32))
    return contexts


def run():
    from chronos import BaseChronosPipeline

    t0 = time.time()
    table = build_feature_table()

    # LightGBMと同一のtest起点/horizon集合を使うため info_level='2' でmetaだけ再利用
    _, y_test, meta_test = make_direct_horizon_samples(table, "test", "2", origin_step=24)
    origins = meta_test["origin"].drop_duplicates().sort_values().reset_index(drop=True)
    print(f"unique test origins: {len(origins)}, total (origin,horizon) rows: {len(meta_test)}")

    print(f"loading {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)

    origin_to_quantiles = {}  # origin -> np.array [MAX_HORIZON, n_quantiles]
    contexts = build_contexts(table, origins)

    for start in range(0, len(contexts), BATCH_SIZE):
        batch_ctx = contexts[start:start + BATCH_SIZE]
        batch_origins = origins.iloc[start:start + BATCH_SIZE]
        q, _mean = pipeline.predict_quantiles(
            inputs=batch_ctx, prediction_length=MAX_HORIZON, quantile_levels=QUANTILES
        )
        q = q.numpy()  # [batch, MAX_HORIZON, n_quantiles]
        for i, o in enumerate(batch_origins):
            origin_to_quantiles[o] = q[i]
        print(f"  batch {start}-{start+len(batch_ctx)}/{len(contexts)} done ({time.time()-t0:.1f}s elapsed)")

    quantile_preds = {q: np.zeros(len(meta_test)) for q in QUANTILES}
    for row_i, (o, h) in enumerate(zip(meta_test["origin"], meta_test["horizon"])):
        qvals = origin_to_quantiles[o][h - 1]  # index h-1 -> t+h
        for qi, q in enumerate(QUANTILES):
            quantile_preds[q][row_i] = qvals[qi]

    scale = naive_seasonal_mae_scale()
    summary, raw = evaluate("chronos_bolt_small", "1", meta_test, y_test.to_numpy(), quantile_preds, naive_scale=scale)

    print(f"\ndone in {time.time()-t0:.1f}s")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
