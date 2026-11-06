# Fugaku Power Forecast

富岳（F-DATA）のジョブスケジューリングログから、消費電力を予測する研究である。
時系列LLM（Chronos-bolt）と、富岳データで学習したMLモデルを比較した。

## 結論

- 短期（1–24h）は Chronos が最良である
- 中期（48–168h）は LightGBM が上回る
- 学習データを約1.4倍に増やすと全体差は縮小したが、Chronos の優位は残った
    - MASE（全体）: Chronos 0.654 / LightGBM 0.771 → Chronos 0.684 / LightGBM 0.724

## 手法

1. ジョブログを1時間グリッドの多変量時系列に集約
2. Chronos / LightGBM / LSTM / TFT を同一条件で比較
3. 変化点検出で安定期とレジーム遷移期に分けて評価
4. SHAP / Attention で covariate の重要度を分析
5. 予測結果を意思決定シミュレーションに接続

## 技術スタック

Python, LightGBM, PyTorch, Chronos, ruptures, SHAP

## 構成

- `scripts/` — 時系列構築・モデル・評価のスクリプト
- `results/` — 評価結果・図
- `docs/` — 詳細な研究記録

## 詳細

- [研究の詳細](docs/details.md)
- [v1 から v2 への再設計](docs/v1-to-v2.md)
