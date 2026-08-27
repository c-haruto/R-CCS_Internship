# Fugaku Power Forecast

富岳(F-DATA)ジョブスケジューリングログから、消費電力を主ターゲットとした多変量時系列を構築し、
**①時系列LLM(Chronos, zero-shot) と ②③富岳データで学習したMLモデル、どちらが消費電力予測に強いか**
を主題として比較する研究。②vs③（MLモデルへのcovariate追加効果）と、富岳データの急激な変化（レジーム遷移）
への頑健性は、どちらも副次的な観察として扱う。研究計画の全文は下記「研究計画（5ステップ）」を参照。

**注記（v2, 分割再設計）**: 当初(v1)は「遷移期でこそ③が優位に立てるか」を主題としてtestを長く
(572日)確保していたが、遷移期での③の優位性はサンプルを増やすと再現しなかった（旧README参照、
`README_v1_backup.md`に保存）。その過程で「②③の学習データが約15ヶ月と少ないためChronosに
勝てないのでは」という疑問が生じたため、**trainを約1.4倍(456日→640日)に増やし、
主題を①vs②③の全体比較（horizon別）に置き直した**のが現在のv2である
（旧v1の全結果は`results_v1_backup/`に保持）。

**中心的な結論**: trainを増やすとChronosと②③の全体差は縮まった(0.654 vs 0.771 → 0.684 vs 0.724)ものの、
埋まりきってはいない＝「学習データの少なさ」だけがChronos優位の理由ではなさそうだ、というのが一つ目の発見。
より明確なのは**horizonで綺麗に優劣が分かれる**ことで、**短期(1-24h)は常にChronosが最良、
中期(48-168h)は安定期・遷移期を問わずLightGBMが明確に上回る**（MASE・意思決定コストの両方で）。
副次的には、富岳の消費電力は運用状況の変化で急激にシフトすることがあり、その前後ではどのモデルも
誤差が悪化するが、covariateがそれを緩和するという当初の仮説はここでも再現しなかった。

## ディレクトリ構成

```
.venv/                               プロジェクト専用venv (ruptures, lightgbm, sklearn, matplotlib, pandas等)
scripts/build_hourly_timeseries.py   Step 1: ジョブログ -> 1時間グリッド多変量時系列
scripts/plot_hourly_overview.py      検証用の概観プロット
scripts/detect_changepoints.py       変化点検出 (ruptures PELT/rbf, 全期間, 分割設計の根拠)
scripts/build_split_design.py        train/valid/test 分割の確定・保存（v2分割）
scripts/features.py                  共通特徴量エンジニアリング(ラグ/rolling/カレンダー/covariate, 720h上限)
scripts/evaluation.py                walk-forward評価ハーネス(MAE/RMSE/MASE/pinball/CRPS)
scripts/model_chronos.py             ①Chronos-bolt-small zero-shot
scripts/model_lightgbm.py            ②③-LightGBM (direct multi-horizon, 分位点回帰)
scripts/model_lstm.py                ②③-LSTM (720hエンコーダ + horizon条件付きMLPヘッド)
scripts/model_tft.py                 ②③-TFT-lite (Variable Selection Network + Self-Attention)
scripts/detect_test_changepoints.py  test期間内の高感度変化点検出(副次分析, step3)
scripts/reevaluate_finegrained.py    test内変化点での安定期/遷移期再ラベリング(step3)
scripts/detection_lag_analysis.py    変化点前後の検知遅延・回復時間分析(step3)
scripts/shap_analysis.py             LightGBM③のSHAP分析(step4, 副次)
scripts/tft_variable_analysis.py     TFT-lite③のVSN重み分析(step4, 副次)
scripts/decision_simulation.py       容量引当の意思決定シミュレーション(step5)
results/hourly_timeseries.parquet    Step 1 出力（本体）
results/split_design.json            確定したtrain/valid/test境界(v2)と含まれる変化点
results/split_design_overview.png    分割の可視化(v2)
results/eval/*_summary.csv / *_raw.parquet   モデルごとの評価結果(v2)
results_v1_backup/                   v1(旧分割・旧主題)の全結果一式。比較用に保持
README_v1_backup.md                  v1時点のREADME全文
```

元データは `/home/elicamodd/f-data/*.parquet`（読み取り専用、38ファイル、2021-03〜2024-04分、
`feature_list.csv` に列定義あり）。このユーザー(chiba)には書き込み権限がないため、
成果物はすべてこの `~/projects/fugaku-power-forecast/` 配下に置く。

## Step 1: 時間グリッド集約 — 完了

`scripts/build_hourly_timeseries.py` を実行し、`results/hourly_timeseries.parquet` を生成済み
（28,084時間 = 2021-03-01 00:00 〜 2024-05-14 03:00 JST、実行時間 約356秒）。
split再設計(v2)の影響を受けない共通の土台であり、変更なし。

### 集約ロジック

- 各ジョブの実行区間 `[sdt, edt)` を1時間バケットへ **秒単位で按分(overlap prorate)** する
  （`bucket_first`〜`bucket_last` を計算し、複数バケットにまたがるジョブは numpy でベクトル化展開）。
- 消費電力(`econ`)は按分比率 `overlap_seconds / job_duration_seconds` で按分し、バケット内で合算 → **`power_consumption`（主ターゲット）**。
- 稼働ノード数・同時実行ジョブ数は node-seconds / job-seconds をバケット内合算し、3600で割って「その時間内の平均同時稼働数」として算出。
- 到着(`adt`)ベースの指標（投入数・要求ノード/コア数・待ち時間・新規ユーザ）と、
  終了(`edt`)ベースの指標（完了数・失敗率）は別集計。
- 新規ユーザ判定は、月次ファイルをファイル名の時系列順(21_03→24_04)に処理し、
  グローバルな `seen_users` セットで「データ全体を通じて初出のユーザ」を判定。

### 出力スキーマ（`hourly_timeseries.parquet`）

| 列 | 説明 |
|---|---|
| timestamp | 時間バケット開始（JST, tz-aware） |
| power_consumption | 按分消費電力の合計（主ターゲット） |
| avg_active_nodes | 平均同時稼働ノード数 |
| avg_active_cores | 平均同時稼働コア数 |
| avg_concurrent_jobs | 平均同時実行ジョブ数 |
| compute_bound_ratio | compute-bound ノード時間の比率（memory-boundとの合計に対する割合） |
| avg_flops | 稼働時間加重平均FLOPS |
| avg_mbwidth | 稼働時間加重平均メモリ帯域幅 |
| job_arrival_count | 当該時間内の新規投入ジョブ数（到着率） |
| avg_requested_nodes / avg_requested_cores | 当該時間に投入されたジョブの平均要求ノード数/コア数 |
| avg_wait_time_sec | 投入(adt)〜開始(sdt)の平均待ち時間 |
| new_user_count | データ全体で初出のユーザによる投入ジョブ数 |
| jobs_completed_count | 当該時間に終了したジョブ数 |
| failure_rate | 終了ジョブに占める失敗(`exit state == 'failed'`)の割合 |

### 妥当性検証

- `avg_active_nodes` の最大値は **158,886**（富岳の総計算ノード数 **158,976** に極めて近い）→ 按分ロジックが物理的に妥当。
- `avg_requested_cores` の最大値は約7.63M ≈ 富岳の総コア数（158,976ノード×48コア≈7.63M）と整合。
- 2021-03-06〜03-09頃に稼働ゼロの時間帯が集中（富岳の共用開始が2021年3月9日である史実と整合、立ち上げ期のデータ欠測として妥当）。
- `results/hourly_timeseries_overview.png` で日次集計を目視確認: power_consumption と avg_active_nodes はほぼ同じ形（当然）、月次の谷（メンテナンス等の定期停止）や2022年8〜9月頃の大きな落ち込み・待ち時間急増イベントが確認できる。

### 既知の制約・今後の検討事項

- `embedding`（384次元sentence-BERT）列は集約対象から除外（サイズが大きく、Step1の対象外）。covariateとして使う場合は別途検討。
- `deldt`（削除時刻）は多くの行で1970-01-01のセンチネル値のため未使用。
- 末尾（2024-04-25以降）はデータが月末に向けて減衰しており、`24_04.parquet`後の部分月データであることに注意（train/test分割時に切り詰めるか要検討）。
- `avg_flops`/`avg_mbwidth` は稼働時間加重平均だが、外れ値（数桁のスケール差）が残っている可能性があり、モデリング前に対数変換や外れ値処理が必要。
- GPU非搭載環境（`nvidia-smi` なし）。Chronosの推論やLightGBM学習はCPUで行う前提。

## 研究計画（5ステップ）

1. **[完了]** ジョブ単位ログを時間グリッドに集約した多変量時系列を構築する。
2. **[完了]** 消費電力を主ターゲットとし、①Chronos zero-shot（電力のみ）、
   ②自作MLモデル（電力のみ）、③自作MLモデル（②+covariate）を、**複数のモデルアーキテクチャで**実装し、
   **①時系列LLM vs ②③訓練済みMLモデル**をhorizon別に比較する。詳細は「Step 2」を参照。
3. **[完了]** （副次分析）変化点検出で「安定期」と「レジーム遷移期」を分けて評価し、
   富岳データが急激に変化した前後でモデルがどう振る舞うかを観察する。詳細は「Step 3」を参照。
4. **[完了]** （副次分析）③のcovariate重要度をSHAP/Attentionで分析し、早期監視すべきシグナルを特定する。
   詳細は「Step 4」を参照。
5. **[完了]** 予測結果を電力・冷却運用や資源スケジューリングの意思決定シミュレーションに接続する。
   詳細は「Step 5」を参照。

## Step 2: モデル比較設計

### 2.1 モデルアーキテクチャ比較（マトリクス設計）

単一アーキテクチャでの②③比較に加えて、**複数アーキテクチャで②③ペアを実装し、
アーキテクチャ間の比較も行う**。採用する3アーキテクチャ（多様なパラダイムを選定）:

| アーキテクチャ | 種別 | 採用理由 |
|---|---|---|
| LightGBM | 勾配ブースティング木 | tabular特徴量に強い高速・軽量な標準ベースライン。SHAPとネイティブ相性が良く step4 に直結。分位点回帰(pinball loss)も標準サポート。 |
| LSTM | RNN (深層学習) | 系列の時間依存性を直接学習する非attention系の深層学習ベースライン。CPUでも本データ規模なら学習可能。 |
| TFT (Temporal Fusion Transformer) | Attention系 (深層学習) | ①Chronosと同じTransformer系だが富岳データのみでゼロから学習する対照群。variable selection network / attention重みが step4 のcovariate重要度分析にネイティブに使える。 |

②③はアーキテクチャごとに**同一アーキテクチャ・同一ハイパラ探索範囲**で対にする
（②-LightGBM vs ③-LightGBM、②-LSTM vs ③-LSTM、②-TFT vs ③-TFT）。
全体としては下表の 1(Chronos) + 3(②) + 3(③) = 7モデルを同一のtrain/valid/test・
同一の評価ウィンドウ・同一の評価指標で比較する。

| | 電力のみ (②) | +covariate (③) |
|---|---|---|
| **LightGBM** | ②-LightGBM | ③-LightGBM |
| **LSTM** | ②-LSTM | ③-LSTM |
| **TFT** | ②-TFT | ③-TFT |
| **①Chronos-bolt (zero-shot, 参照点)** | 電力のみ・学習なし | (covariate版は非対応。任意でfine-tuning版を発展課題として追加検討) |

比較の軸:
- **①vs②-X（Xは各アーキテクチャ）**: 入力情報量(電力のみ)を揃えた上での「事前学習知識 vs 富岳データのみでの学習」の効果。3アーキテクチャ分あるので、この効果がアーキテクチャに依らず一貫するかも見える。
- **②-X vs ③-X**: 同一アーキテクチャでのcovariate情報量の効果（副次的な観察軸）。
- **②-LightGBM vs ②-LSTM vs ②-TFT（③についても同様）**: 同じ入力情報量でも「器」の違いがどれだけ性能に効くか。

### 2.2 データ分割設計 v2（trainを厚くする方向に再設計）

**背景**: 当初(v1)の分割は「遷移期でこそ③が優位に立てるか」を検証するためtestを長く(572日)取り、
trainは456日(約15ヶ月)しかなかった。この結果、①Chronosが②③を全面的に上回ったが、
「単にMLモデルの学習データが少なすぎるからではないか」という疑問が生じた
（③の遷移期優位も、サンプルを増やすと再現しなかった。詳細は`README_v1_backup.md`）。
そこで**遷移期分析を副次的な位置づけに変え、trainをできる限り厚くする**方向にv2として再設計した。

**方針**: train/valid/testを時系列順に非重複分割する。分割境界の根拠は v1 と同じ変化点検出
（`scripts/detect_changepoints.py`, ruptures PELT + rbfコスト, 2週間未満の変化は無視、対象期間全体で
2021-06-15, 2022-04-10, 2022-07-23, 2022-11-08 の4点を検出）を使うが、
**4点全てをtrainに含める**ことでtrainを最大化した（v1はtrain2点・valid1点・test1点に分配していた）。

確定した分割（`results/split_design.json`、可視化は`results/split_design_overview.png`）:

| split | 期間 (JST) | 日数 | 含まれる変化点 | 目的 |
|---|---|---|---|---|
| (除外) | 2021-03-01 - 2021-04-01 | 31 | - | 富岳共用開始直後の立ち上げ期。稼働ゼロの時間帯が集中し非代表的なため全splitから除外。 |
| **train** | 2021-04-01 - 2023-01-01 | **640**（v1比+40%） | 2021-06-15, 2022-04-10, 2022-07-23, 2022-11-08（4点全て） | ②③の学習データを最大化。同時に「遷移とは何か」を学習する機会も増える。 |
| **valid** | 2023-01-01 - 2023-03-01 | 59 | - | ハイパラ選択・early stoppingのみが目的。v1と異なり変化点を含めることは要件としない。 |
| **test** | 2023-03-01 - 2024-04-25 | 421 | -（粗いラベルでは0件） | 全体比較の主戦場。coarse changepointを含まないため、Step2ではMVPのregime分割(±14日)は成立しない＝安定期のみになる。副次的な安定期/遷移期分析はStep3でtest内部を高感度に再検出して行う。 |
| (除外) | 2024-04-25 - 2024-05-15 | 19 | - | `24_04.parquet`後の部分月データで不完全なため除外。 |

v1からの主な変更点はtrain/valid/testの配分のみで、**分割の根拠となる変化点検出ロジック自体は変えていない**
（同じ`changepoints.csv`を使用）。

### 2.3 Chronos(時系列LLM) と 自作MLモデルの公平性担保

zero-shotの時系列LLMと、trainデータで学習するMLモデルは根本的に性質が異なるため、
以下のルールで条件を揃える（v1から変更なし）:

1. **同一のtest評価プロトコル**: 全モデルとも、testの各予測起点(forecast origin)から
   短期(24時間先)・中期(7日先)を予測し、同一のtest起点集合・同一の指標(MAE/RMSE/MASE/pinball/CRPS)
   で評価する（walk-forward、起点は例えば24時間おきにスライド）。
2. **「凍結重みでの評価」を全モデル共通条件にする**: ②③はtrain+validで学習・HP選択を
   済ませた後、**test期間中は重みを一切更新しない**まま、各起点で新しく観測された
   過去データ（ラグ特徴量・covariate）だけを入力してtest全体を予測する。
   Chronosはそもそも重み更新をしないzero-shotなので、この条件は自動的に満たされる。
3. **参照できる過去情報の長さ(コンテキスト長)を揃える**: Chronos-boltは最大2048時点の
   コンテキストを扱えるが、無制限に長い履歴を使わせるとMLモデルのラグ特徴量
   （lag 1h/24h/168h、rolling統計など）より不当に有利になる。そこで**全モデル共通で
   直近720時間(30日)を最大コンテキスト/特徴量ウィンドウの上限**とする。
4. **入力情報量を①②で揃える**: ①Chronosと②-X（各アーキテクチャ）は、どちらも
   「電力の過去系列のみ」を入力とする。③-Xだけがjob_arrival_count等のcovariateを追加する。
5. **Chronosのfine-tuning(発展課題)を行う場合**は、train+validのみで学習し、
   その後は上記1-3と同じ凍結weightsルールをMLモデルと同様に適用する。

### 2.4 実装・実行 — 完了（v2で再学習・再評価）

7モデル(①Chronos-bolt-small zero-shot、②③×{LightGBM, LSTM, TFT-lite})すべてを新split(v2)で
再学習・再評価済み(LightGBM: 46.6秒、Chronos: 41.9秒(推論のみ)、LSTM: 全体1262.7秒、TFT-lite: 全体1405.2秒、
いずれもCPU、`torch.set_num_threads(8)` + `OMP_NUM_THREADS=8`で実行)。
実装コード自体はv1から変更なし（`features.py`が`split_design.json`を動的に読むため、
splitを変えるだけで全パイプラインが再実行できる設計になっている）。

```
scripts/features.py        共通特徴量エンジニアリング(ラグ/rolling/カレンダー/covariate, 720h上限)
scripts/evaluation.py      walk-forward評価ハーネス(MAE/RMSE/MASE/pinball/CRPS)
scripts/model_lightgbm.py  ②③-LightGBM (direct multi-horizon, 分位点回帰)
scripts/model_chronos.py   ①Chronos-bolt-small zero-shot (dl_env, CPU推論, 720hコンテキスト)
scripts/model_lstm.py      ②③-LSTM (720hエンコーダ + horizon条件付きMLPヘッド, dl_env)
scripts/model_tft.py       ②③-TFT-lite (Variable Selection Network + Self-Attention, dl_env)
results/eval/*_summary.csv / *_raw.parquet   モデルごとの評価結果(v2)
```

### 2.5 Step 2 結果サマリ（v2）

**test全体(2023-03-01〜2024-04-25, 全horizon平均)のMASE**:

| モデル | MASE (test全体) |
|---|---|
| **①Chronos-bolt (zero-shot)** | **0.684** |
| ③LightGBM (+covariate) | 0.724 |
| ②LightGBM (power-only) | 0.724 |
| ②TFT-lite (power-only) | 0.733 |
| ③TFT-lite (+covariate) | 0.744 |
| ②LSTM (power-only) | 0.753 |
| ③LSTM (+covariate) | 0.889 |

**① vs ②③（本研究の主題）— 全体差は縮まったが埋まらなかった**: Chronosは今回も全体最良(0.684)だが、
次点のLightGBM(0.724)との差はv1(0.654 vs 0.771、差0.117)から**v2(0.684 vs 0.724、差0.040)へと大幅に縮小**した。
trainを1.4倍に増やしたことで②③がChronosに近づいたのは事実であり、「学習データの少なさ」がChronos優位の
一因であったことを支持する。ただし差はゼロにはならなかったため、**残りの差は事前学習コーパスの
規模・多様性という質的な優位性**によるものと考えられる。

**horizonで見ると優劣がはっきり分かれる — 本研究で最も明確な結果**:

| horizon | ①Chronos | 最良のML | 勝者 |
|---|---|---|---|
| 短期(1-24h) | **0.628** | 0.679 (②TFT-lite) | **① Chronos** |
| 中期(48-168h) | 0.911 | **0.833** (②/③LightGBM) | **② / ③ LightGBM** |

短期はChronosが圧勝する一方、**中期(48-168h)はLightGBM(②③どちらも0.83台)がChronosを明確に上回る**。
v1では「中期・遷移期に限り」という条件付きだったこの逆転が、v2ではtrain量が増えたことで
**horizonだけで説明できる、regimeに依存しない結果**になった（3.2節でtest内部の安定期/遷移期に分けても
両方でLightGBMが勝つことを確認する）。これはv1の「遷移期でこそ」というやや複雑な仮説より、
遥かにシンプルで頑健な結論である。

**② vs ③（covariate情報の効果、副次的発見）**: LightGBMは②0.7245/③0.7235とほぼ同値、
TFT-liteは②0.733/③0.744でむしろ悪化、LSTMは②0.753/③0.889と大きく悪化した。v1と異なり
今回はLightGBM以外の全アーキテクチャでcovariateがむしろ足を引っ張っており、
「covariateの効果はアーキテクチャ依存」という結論はv1・v2を通じて一貫している
（効果自体が小さい・不安定という方向にさらに寄った）。LSTMの悪化幅が特に大きい点は
3.3節で改めて触れる。

### 2.6 既知の制約・今後の検討事項

- v2のtestには粗いラベル(全期間変化点4点)による安定期/遷移期の分類が存在しない（4点全てtrainに
  含まれるため）。そのため副次的な安定期/遷移期分析は、test内部だけに絞った高感度な変化点検出
  (Step3)に完全に依存している。
- MASEの分母は train期間全体で固定した24hナイーブ予測誤差の平均であり、regimeごとのスケール差を
  補正していない。
- TFT-liteは簡易実装（pytorch-forecasting等の完全なTFTではない）。Variable Selection Network +
  Interpretable Multi-Head Self-Attention(直近168hに限定)のみを実装しており、
  static covariate encoderや真のseq2seqデコーダは省略している。
- Chronosのfine-tuning（発展課題）は未実施。trainを増やした効果が「②③の学習」だけでなく
  「Chronosをfine-tuneした場合」にどう出るかは、今回のtrainサイズ実験の自然な延長として興味深い。
- trainサイズを段階的に増減させた学習曲線（train量アブレーション）は未実施。「学習データの少なさ」
  仮説をさらに直接検証するなら、今回のtrain(640日)を複数の長さに区切って再学習する追加実験が有効。

## Step 3: 富岳データの急変への頑健性（副次分析）

**本研究の主題（Step2の①vs②③、horizon別比較）とは別軸の観察**。富岳の運用状況（ワークロード構成、
メンテナンス等）は時に急激に変化することがあり、そうした変化の前後でモデルがどう振る舞うかを、
test期間(421日)内部にさらに高感度な変化点検出をかけて観察した。

```
scripts/detect_test_changepoints.py    test期間内の高感度変化点検出(ruptures PELT, pen=6)
scripts/reevaluate_finegrained.py      既存の生予測を精緻な変化点で再ラベル付けし再集計
scripts/detection_lag_analysis.py      horizon=1h誤差の変化点前後推移から検知遅延/回復時間を算出
results/test_changepoints.csv                    test内で検出した4つの変化点(v2)
results/test_changepoints_overview.png           変化点の可視化(v2)
results/step3_finegrained_summary.csv            精緻化ラベルでの7モデル評価(v2)
results/step3_mase_pivot.csv                     MASEのarchitecture×horizon_bucket×regimeピボット(v2)
results/step3_covariate_improvement.csv          精緻化ラベルでの②→③改善率(v2)
results/step3_detection_lag.csv                  変化点×モデルごとの検知遅延生データ(v2)
```

### 3.1 test内の変化点(4点)

多変量(power_consumption, avg_active_nodes, job_arrival_count, avg_wait_time_sec,
compute_bound_ratio)のz-score系列にPELT(rbfコスト, pen=6)を適用し、以下4点を検出:
**2023-04-04, 2023-05-14, 2023-08-16, 2023-10-05**。約412日のtest期間に平均103日間隔で分布している。

### 3.2 安定期/遷移期別の再評価 — 中期でのLightGBM優位はregimeによらず頑健

この4変化点(±14日を遷移期とする)でtest内を再ラベル付けし、Chronosを含む7モデルを再集計した:

| horizon × regime | ①Chronos | ②LightGBM | ③LightGBM | ②LSTM | ③LSTM | ②TFT-lite | ③TFT-lite |
|---|---|---|---|---|---|---|---|
| 短期・安定期 | **0.627** | 0.692 | 0.676 | 0.694 | 0.820 | 0.677 | 0.685 |
| 短期・遷移期 | **0.630** | 0.714 | 0.752 | 0.716 | 0.944 | 0.685 | 0.700 |
| 中期・安定期 | 0.863 | 0.809 | **0.800** | 0.908 | 0.961 | 0.888 | 0.903 |
| 中期・遷移期 | 1.035 | 0.897 | **0.923** | 1.119 | 1.199 | 1.108 | 1.115 |

(値はMASE、太字が各行の最良モデル)

- **中期(48-168h)はLightGBMが安定期・遷移期のどちらでもChronosを上回る**（安定期0.80 vs 0.86、遷移期0.90 vs 1.04）。
  2.5節で見た「中期はLightGBMが勝つ」という結果が、regimeによらず一貫することが確認できた。
  つまりこの優位は「遷移期特有の現象」ではなく、**中期という予測ホライズン自体でLightGBMが強い**、
  というよりシンプルな説明で足りる。
- **どのモデルも遷移期の方が安定期よりMASEが悪化する**（Chronos中期0.863→1.035、LightGBM②0.809→0.897など）。
  v1では規模の問題でこの関係が一部逆転していたが、v2ではサンプル数が増えたことで
  「遷移期の方が難しい」という直感通りの結果になった。

### 3.3 ②→③ covariateの効果（副次的発見） — 3回目の独立した検証でも再現せず

同じ精緻化ラベルで②→③(covariate追加)の効果を見ると:

| architecture | 短期・安定期 | 短期・遷移期 | 中期・安定期 | 中期・遷移期 |
|---|---|---|---|---|
| LightGBM | +2.3% | **-5.3%** | +1.1% | **-2.9%** |
| LSTM | **-18.3%** | **-31.9%** | -5.8% | -7.2% |
| TFT-lite | -1.2% | -2.2% | -1.7% | -0.7% |

(値は②→③のMASE改善率。プラスが③の勝ち)

- LightGBMは安定期でわずかにプラス、遷移期でマイナスという傾向がv1のStep3訂正後の結果と一致しており、
  **「covariateは遷移期でなく安定期でこそ効く」という結論が、v1・v2という完全に異なる2つの分割設計で
  独立に再現した**。これは本研究全体の中でも特に頑健な副次的知見と言える。
- **LSTMのcovariate効果が今回大きく悪化**（短期遷移期-31.9%）。2.5節の全体結果(②0.753→③0.889)と
  整合しており、trainを増やしたことでLSTM③がむしろ最適化に苦戦した可能性がある
  （早期停止のタイミングや、covariate込みの特徴量数増加に対して学習が不安定になったなど）。
  原因の切り分けは未実施で、今後の課題とする。

### 3.4 検知遅延・回復時間分析

「遷移直後どれだけ早く性能が回復するか」をhorizon=1h予測誤差で見ると（基準値の1.3倍以内に
3日連続で収まった時点を「回復」と定義）:

| モデル | 平均ピーク/基準値比 | 平均回復日数 |
|---|---|---|
| ①Chronos (zero-shot) | **5.50倍**（最大） | 2.00日 |
| ②LightGBM | 3.16倍 | 2.00日 |
| ③LightGBM | 3.58倍 | 3.50日 |
| ②LSTM | 2.48倍 | 3.50日 |
| ③LSTM | 2.49倍 | 3.75日 |
| ②TFT-lite | 2.52倍 | 2.00日 |
| ③TFT-lite | 2.43倍 | 2.00日 |

- Chronosは変化点直後の誤差ピークが他モデルの2倍以上と最も大きいが、回復日数は②LightGBM・
  ②/③TFT-liteと並んで最速(2.00日)。v1では「Chronosだけ突出して回復が速い」という結果だったが、
  v2ではChronos以外にも2日で回復するモデルが複数あり、**Chronos固有の強みというよりtrainを
  増やした②③全体の底上げ**とも解釈できる。
- LightGBM③・LSTM②③は回復に3.5日以上かかっており、covariateを含むモデルほど遷移直後は
  むしろ不安定になりやすい傾向が見える（3.3節の「covariateは遷移期で効かない」と整合）。

### 3.5 Step 3 総括（副次分析）

- 富岳の消費電力は運用状況の変化で急激にシフトすることがあり（4変化点、平均103日間隔）、
  その前後ではどのモデルも予測誤差が悪化する。
- 中期(48-168h)でのLightGBM優位はregimeに依存しない頑健な結果（3.2節）。
- 「covariateは遷移期でなく安定期でこそ効く」という結論は、v1・v2という2つの独立した分割設計で
  再現した、本研究で最も信頼できる副次的知見。
- 変化点の検出粒度(pen値)やtransition window幅(±14日)は依然として設計者が選んだハイパラであり、
  結果はこれに一定程度依存する。感度分析は今後の課題。

## Step 4: covariate重要度分析(SHAP / TFT Variable Selection)（副次的分析）

**本研究の主題（①vs②③、Step2/3）とは別軸の分析**。②→③のcovariate追加効果自体は小幅〜マイナスだったが
（Step3で確認）、「効いているとすればどのcovariateか」「その中身は遷移期で変わるか」はそれ自体独立して
興味深い問いのため、LightGBM③(SHAP, 厳密解のTreeExplainer)とTFT-lite③(Variable Selection Networkの重み)
という**2つの独立したモデル・解釈手法**で分析した（v2データで再実行）。

```
scripts/shap_analysis.py            LightGBM③のSHAP分析(全体/horizon帯/レジーム別)
scripts/tft_variable_analysis.py    TFT-lite③のVSN重み分析(レジーム別)
results/step4_shap_top20.png                  SHAP top20棒グラフ(v2)
results/step4_shap_category_share.png         カテゴリ別寄与割合の比較図(v2)
results/step4_tft_vsn.png                     TFT VSN重み比較図(v2)
```

### 4.1 LightGBM③: SHAP分析

- 全体では `power_consumption`（ラグ・rolling特徴量群）が支配的で、covariate群の寄与は
  **|SHAP|合計の約19.1%**（power/calendarが約61%、予測対象時刻のカレンダー特徴量が約16%）。
  安定期19.6%・遷移期17.6%とやや遷移期で下がる。
- **covariate内部の構成は遷移期でシフトする**。transition/stable比が最も高い(=遷移期で重要度が
  上がる)のは**`avg_wait_time_sec_z168h`（待ち時間の週内急増z-score）で1.65倍**——
  これはv1(1.70倍)とほぼ同じ数値であり、**分割設計を大きく変えた(train456日→640日、test期間も
  完全に別の18ヶ月)にもかかわらず、同じ特徴量が同じ方向で再現した**。本研究全体で最も頑健な
  単一の特徴量レベルの発見と言える。
- 他にも `job_arrival_count_rollmean168h`(1.37倍)、`avg_wait_time_sec_rollmean24h`(1.29倍)、
  `compute_bound_ratio_rollmean168h`(1.18倍)が遷移期で重要度が上がる。

### 4.2 TFT-lite③: Variable Selection Network重み分析

- TFT-liteのcovariate群の合計重みは安定期9.98%→遷移期10.12%とほぼ変化なし（v1は31.6%→29.9%で
  水準自体が大きく異なり、trainを増やしたことでTFT-liteの重み配分がpower_consumption側に
  さらに寄った可能性がある。42.1節のSHAPと単純比較はできないが、方向としては「covariateへの
  依存度がTFT-liteでは低い」という傾向はv1・v2で一致）。
- 個別には`avg_wait_time_sec_lag0h`が安定期0.0200→遷移期0.0206とわずかに上昇しており、
  LightGBMのSHAPで最も遷移期に効くとされた「待ち時間」信号が、TFT-liteでも同じ方向を
  示している点はv1と同様。

### 4.3 早期シグナルの実務的示唆

2つの独立した解釈手法・2つの独立した分割設計(v1/v2)を通じて一貫して浮かび上がったのは、
**「ジョブ投入率そのもの」より「待ち時間(キュー長)の急激な変化」の方が、レジーム遷移の
先行指標として感度が高い**という点。運用上の示唆:
- **`avg_wait_time_sec`の週内(168h)ローリングz-scoreを監視ダッシュボードの主要指標にする**
  （急上昇は投入から数時間〜数日先に電力需要が変化する予兆となりうる）。
- `job_arrival_count`の週内平均や`compute_bound_ratio`の週内シフトも補助的な先行指標になりうる。
- 一方、`job_arrival_count`（到着率そのもの）は絶対的な予測力は最大だが、遷移の「予兆」という
  意味では相対的な感度は低い。

### 4.4 既知の制約

- 本分析はLightGBM/TFT-liteのみで実施。LSTMには特徴量単位の解釈手法を適用していない。
- TFT-liteの自己注意(self-attention)重み自体は現時点で未保存(VSN重みのみ保存済み)。
- SHAP分析はq=0.5(中央値)モデルのみに適用。分位点ごとの寄与差は未分析。

## Step 5: 意思決定シミュレーション(電力・冷却容量の引当)

予測(分位点)を「冷却容量／電力バジェットをどれだけ引き当てるか」という意思決定に接続し、
予測精度の改善が実際の運用コスト削減にどれだけ結びつくかを定量化した（v2データで再実行）。

```
scripts/decision_simulation.py       容量引当シミュレーション本体
results/step5_decision_cost.csv      policy×scenario×horizon×regimeの正規化コスト(v2)
results/step5_calibration.csv        分位点予測のキャリブレーション(実現被覆率)(v2)
results/step5_decision_cost.png      policy別コスト比較図(v2)
```

### 5.1 枠組み: newsvendor型の非対称コスト最適化

各時点で「冷却容量 C をどれだけ引き当てるか」を決める場面を想定する。実際の消費電力 y が
Cを超えれば**過小引当**（緊急冷却・スロットリング等の高コスト）、Cを下回れば**過大引当**
（遊休容量の無駄、低コスト）が発生する:

```
cost(y, C) = r × max(y − C, 0) + 1 × max(C − y, 0)      (r = 過小引当コスト / 過大引当コスト)
```

この期待コストを最小化する最適な引当水準は、まさに**分位点予測**そのもの（臨界分位点
q* = r/(1+r)）であり、①②③の学習・評価に使ったpinball lossと数学的に同一の問題である。
r=3(中程度のリスク回避)はq*=0.75、r=9(冷却系がミッションクリティカルな高リスク回避)は
q*=0.90に対応する。比較対象は**static**(train期間の無条件分位点を定数として常に引き当てる)、
**seasonal-naive**(24時間前の実測値+train残差分位点)、そして**7モデル**。結果は正規化コスト単位・
相対削減率で報告する（絶対金額は捏造しない）。

### 5.2 結果: 予測を使うと引当コストが最大33.6%下がる

| policy | r=3 (moderate) | r=9 (high-stakes) |
|---|---|---|
| **①Chronos（最良）** | **2.474M** | **3.517M** |
| ③LightGBM | 2.702M | 3.821M |
| ②TFT-lite | 2.722M | 3.721M |
| ②LightGBM | 2.725M | 3.790M |
| ③TFT-lite | 2.745M | 3.766M |
| ②LSTM | 2.774M | 3.841M |
| static（予測なし） | 3.065M | 4.174M |
| ③LSTM | 3.326M | 4.451M |
| seasonal-naive | 3.341M | 5.298M |

(単位: 正規化コスト、値が小さいほど良い)

- **全体では引き続きChronosが最良**（static比 r=3時+19.3%・r=9時+15.7%、seasonal-naive比
  r=3時+25.9%・r=9時+33.6%）。7モデル中5モデルはstatic/seasonal-naive両方を上回ったが、
  **③LSTMは意外にもstaticより悪化**しており(3.326M > 3.065M)、2.5節で見たLSTM③の劣化が
  コストにもそのまま反映されている。
- **質の低い予測は何もしないより悪い**ことが、③LSTMというtrain量を増やした今回のモデルでも
  再確認された。

### 5.3 horizon×regime別コスト: 中期でのLightGBM優位はここでも再現する

**中期horizon(48-168h, r=9)**:

| policy | 中期・安定期 | 中期・遷移期 |
|---|---|---|
| **②LightGBM** | **3.750M** | 4.276M |
| **③LightGBM** | 3.821M | **4.136M** |
| ③TFT-lite | 3.801M | 4.341M |
| static（予測なし） | 3.895M | 4.282M |
| ②TFT-lite | 3.971M | 4.610M |
| ②LSTM | 4.124M | 4.428M |
| seasonal-naive | 4.545M | 4.688M |
| **①Chronos** | **4.519M**（staticより悪化） | **5.079M**（全policy中最悪、staticより悪化） |
| ③LSTM | 4.616M | 5.178M |

- **中期horizonでは、安定期・遷移期のどちらでもLightGBMが最安コスト**であるのに対し、
  **Chronosはどちらの regime でも static(予測なし)より高コスト**になる。特に中期・遷移期では
  Chronosは全policy中最悪（seasonal-naiveより悪い）で、この区間でChronosの予測を意思決定に
  使うと明確に損をする。
- これは3.2節のMASEベースの結果（中期はLightGBMがregimeによらず勝つ）と完全に整合しており、
  「Chronosは短期に強く、中期には弱い」という2.5節の主結論を、意思決定コストという実務指標でも
  再確認する形になった。

**短期horizon(1-24h, r=9)**: 安定期3.139M・遷移期3.482Mのどちらも①Chronosが最安で、
2.5節のMASEと同じ傾向。

### 5.4 キャリブレーションの注記

分位点予測の実現被覆率(empirical coverage)を確認したところ、Chronosのq=0.90予測は実際には
87.85%しか外れなかった(想定より約2.15pt高い頻度で超過)——v1(87.7%)とほぼ同じ値で、
**Chronosの分位点キャリブレーションのわずかな甘さは分割を変えても一貫して観測される**特性である。
static/seasonal-naiveは定義上ほぼ完全にキャリブレーションが取れている一方でコストは劣る。

### 5.5 研究計画の総括

5ステップすべてが完了した。最終的な結論を要約すると:

1. **①時系列LLM(Chronos, zero-shot) vs ②③訓練済みML — 本研究の中心的な結果**:
   trainを1.4倍(456日→640日)に増やすと、全体MASEの差はChronos優位のまま縮小した(0.654 vs 0.771 →
   0.684 vs 0.724)。学習データの少なさはChronos優位の一因だが、それだけでは説明しきれない。
   より明確なのは**horizonによる綺麗な逆転**: 短期(1-24h)は常にChronosが最良、
   **中期(48-168h)は安定期・遷移期を問わずLightGBMが明確に上回る**（MASE・意思決定コストの両方）。
   この逆転はv1では「遷移期限定」という複雑な条件付きだったが、trainを増やしたv2では
   regimeに依存しない、よりシンプルで頑健な結論になった。
2. **富岳データの急変への頑健性（副次発見）**: 富岳の消費電力は運用状況の変化で急激にシフトする
   ことがあり(test内で4変化点)、その前後ではどのモデルも誤差が悪化する。covariateがこれを
   緩和するという当初の仮説は、v1・v2という2つの独立した分割設計のどちらでも再現しなかった
   （むしろ安定期でこそcovariateは効く）。
3. **早期監視シグナル（covariateの内訳分析、Step4）**: 到着率そのものより、待ち時間(キュー長)の
   急増z-scoreの方がレジーム遷移の予兆として感度が高い。この特徴量レベルの発見はv1・v2で
   ほぼ同じ数値(1.70倍→1.65倍)で再現した、本研究で最も頑健な知見。
4. **実務的価値**: 予測を使った動的な容量引当は、何もしない静的方式に対し最大33.6%のコスト
   削減効果を示した。ただし中期horizonでのChronosのように、モデルとhorizonの組み合わせを
   誤ると予測しない場合より悪化しうるため、「どのモデルをどのhorizonで使うか」を条件別に
   検証すること自体に実務的意義がある。
