"""
発展分析: avg_active_nodes予測でのLightGBM②③（元のmodel_lightgbm.pyと同一設計・同一ハイパラ）。
"""
import os
import time

import lightgbm as lgb
import pandas as pd

from features_nodes import build_feature_table, make_direct_horizon_samples
from evaluation_nodes import evaluate, naive_seasonal_mae_scale, QUANTILES

RESULTS = os.path.expanduser("~/projects/fugaku-power-forecast/results")
MODELS_DIR = os.path.expanduser("~/projects/fugaku-power-forecast/models")
os.makedirs(MODELS_DIR, exist_ok=True)

LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)


def train_one_quantile(X_train, y_train, X_valid, y_valid, q):
    model = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model


def run(info_level):
    t0 = time.time()
    table = build_feature_table()

    X_train, y_train, _ = make_direct_horizon_samples(table, "train", info_level, origin_step=6)
    X_valid, y_valid, _ = make_direct_horizon_samples(table, "valid", info_level, origin_step=6)
    X_test, y_test, meta_test = make_direct_horizon_samples(table, "test", info_level, origin_step=24)

    print(f"[nodes info_level={info_level}] train={X_train.shape} valid={X_valid.shape} test={X_test.shape}")

    quantile_preds = {}
    importances = None
    for q in QUANTILES:
        m = train_one_quantile(X_train, y_train, X_valid, y_valid, q)
        quantile_preds[q] = m.predict(X_test)
        if q == 0.5:
            importances = pd.Series(m.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print(f"  q={q} trained, best_iteration={m.best_iteration_}")

    model_name = f"lightgbm_info{info_level}_nodes"
    scale = naive_seasonal_mae_scale()
    summary, raw = evaluate(model_name, info_level, meta_test, y_test.to_numpy(), quantile_preds, naive_scale=scale)

    importances.head(30).to_csv(os.path.join(MODELS_DIR, f"{model_name}_top30_importance.csv"))

    print(f"\n[nodes info_level={info_level}] done in {time.time()-t0:.1f}s")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    all_summaries = []
    for info_level in ["2", "3"]:
        all_summaries.append(run(info_level))
    combined = pd.concat(all_summaries, ignore_index=True)
    os.makedirs(os.path.join(RESULTS, "eval_nodes"), exist_ok=True)
    combined.to_csv(os.path.join(RESULTS, "eval_nodes", "lightgbm_combined_summary.csv"), index=False)
    print("\n=== combined ②③ LightGBM (nodes) summary ===")
    print(combined.to_string(index=False))
