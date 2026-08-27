"""
発展分析: avg_active_nodes予測でのChronos-bolt zero-shot（元のmodel_chronos.pyと同一設計）。
実行はdl_env: ~/dl_env/bin/python scripts/model_chronos_nodes.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features_nodes import build_feature_table, make_direct_horizon_samples, MAX_LOOKBACK, HORIZONS, TARGET_COL
from evaluation_nodes import evaluate, naive_seasonal_mae_scale, QUANTILES

MODEL_ID = "amazon/chronos-bolt-small"
MAX_HORIZON = max(HORIZONS)
BATCH_SIZE = 32


def build_contexts(table, origins):
    contexts = []
    for o in origins:
        window = table.loc[:o, TARGET_COL].iloc[-MAX_LOOKBACK:]
        assert len(window) == MAX_LOOKBACK, f"insufficient context at {o}: {len(window)}"
        contexts.append(torch.tensor(window.to_numpy(), dtype=torch.float32))
    return contexts


def run():
    from chronos import BaseChronosPipeline

    t0 = time.time()
    table = build_feature_table()

    _, y_test, meta_test = make_direct_horizon_samples(table, "test", "2", origin_step=24)
    origins = meta_test["origin"].drop_duplicates().sort_values().reset_index(drop=True)
    print(f"unique test origins: {len(origins)}, total (origin,horizon) rows: {len(meta_test)}")

    print(f"loading {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)

    origin_to_quantiles = {}
    contexts = build_contexts(table, origins)

    for start in range(0, len(contexts), BATCH_SIZE):
        batch_ctx = contexts[start:start + BATCH_SIZE]
        batch_origins = origins.iloc[start:start + BATCH_SIZE]
        q, _mean = pipeline.predict_quantiles(
            inputs=batch_ctx, prediction_length=MAX_HORIZON, quantile_levels=QUANTILES
        )
        q = q.numpy()
        for i, o in enumerate(batch_origins):
            origin_to_quantiles[o] = q[i]
        print(f"  batch {start}-{start+len(batch_ctx)}/{len(contexts)} done ({time.time()-t0:.1f}s elapsed)")

    quantile_preds = {q: np.zeros(len(meta_test)) for q in QUANTILES}
    for row_i, (o, h) in enumerate(zip(meta_test["origin"], meta_test["horizon"])):
        qvals = origin_to_quantiles[o][h - 1]
        for qi, q in enumerate(QUANTILES):
            quantile_preds[q][row_i] = qvals[qi]

    scale = naive_seasonal_mae_scale()
    summary, raw = evaluate("chronos_bolt_small_nodes", "1", meta_test, y_test.to_numpy(), quantile_preds, naive_scale=scale)

    print(f"\ndone in {time.time()-t0:.1f}s")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
