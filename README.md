# TSV-Merge：モデルマージの比較・研究用リポジトリ

同じ基準モデルから作られた学習済みモデル群をマージし、共通の評価処理で比較します。

```text
入力を準備・検査 → 手法でマージ → 検証データで設定を選択
               → 選択時と同じ重みをテスト評価 → 条件・結果・診断を保存
```

[Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors) の独立した研究・再現forkです。公式リポジトリではありません。数値の再現基準は upstream commit `a6c97188f7aa0bb20753de4f2939be89dd64f1c3` です。

- [変更前のREADME](README.original.md)
- [過去の8タスク再現記録](docs/reproduction.md)
- [過去の5手法比較記録](docs/five_method_comparison.md)
- [変更時のルール](AGENTS.md)

## 1．対象と実装の区分

OpenAI CLIPの **ViT-B/32・ViT-B/16・ViT-L/14** と、明示的な **8・14・20タスク**の集合に対応します。入力はFull Fine-tuning済みのチェックポイントです。学習処理全体の再設計は今回の対象外です。

| 手法 | `method=` | 数値実装の基準 |
|---|---|---|
| Weight Averaging | `weight_average` | 元実装の `base + sum(delta)/T` の計算順序 |
| Task Arithmetic | `task_arithmetic` | 全体の成分選択、差分の和、係数α |
| TIES-Merging | `ties` | TSV-M元実装の全モデル閾値・符号規則 |
| Consensus TA | `consensus_ta` | 検証でTALLマスクを選び、全タスク共通の差分を作る |
| TSV-M | `tsvm` | 固定コミットのランク・固定スロット・ゼロ埋め |
| Iso-CTS | `iso_cts` | 公式コード `d34010af8ba129ef4d17a91ca9943b413fe06f8c` |
| DC-Merge | `dc_merge` | 公式FFTコード `089af77a378562272461e0f041217ec440a4a230` |
| SERA | `sera` | 論文 `2609.24517v1` に基づくローカル実装 |
| GSE-TSV | `gse_tsv` | `GlobalRawEnergyTopR`に対応。v2では閾値判定を実行デバイス上で行う |
| Zero-shot（補助基準） | `zeroshot` | 基準重みをそのまま使う |

各手法は共通の画像エンコーダを1つ返します。タスクが既知の評価条件で、タスク別分類ヘッドを接続します。

AdaptiveTSV、部分空間版、正規化エネルギー版などは `method=experimental/...` で利用できます。**研究用エネルギーランクの閾値1における上限処理だけは修正し、v2として区別**しています。旧ソースは `archive/legacy_code/` に保存しています。保護対象4手法の数値規則は維持します。

精査によるv2修正では、GSE-TSV・SERAのエネルギー判定を `runtime.device` 上で行い、CPU退避による閾値境界の変化を防いでいます。乱数を使う研究用 `dummy` も、準備処理や分母キャッシュの有無から独立したseed付き生成器へ変更しました。実装版は結果へ記録し、過去の結果は書き換えません。

評価adapterのv3では、OpenAI CLIPと同じQuickGELUを明示してモデルを構築します。v2以前の `pretrained=None` だけの構築は通常のGELUになり、同じ重みでも推論結果が変わる問題がありました。v3は評価器の識別情報とソースのチェックサムが異なるため、旧分母を再利用せず、旧結果との同条件比較も拒否します。過去の結果は保持します。

旧 `tall_mask`・`mag_masking`・`TSVC` はタスクごとにモデルが異なる実験です。今回は過去コード・結果を保存する範囲とし、現行CLIからの実行は受け付けません。

**実装と精度の再現確認は別です。** 81条件の設定を列挙できることは、81条件を実データで評価済みという意味ではありません。各実行の条件・実装版・未確認の由来は結果に記録します。配布データ・重みはGitに含みません。

## 2．環境と入力の準備

既存の `environment.yml` を利用します。新基盤の実験にW&B接続は必要ありません。図の出力にはMatplotlibが必要で、環境定義に含めています。

```bash
micromamba env create -f environment.yml
micromamba run -n tsv python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

以下のコマンドはリポジトリのルートで実行してください。CPUを選ぶ場合は、新CLIでは `runtime.device=cpu` を明示します。メモリ不足時に精度やアルゴリズムを自動変更しません。

既定の入力は次の配置です。

```text
models/checkpoints/ViT-B-32/
├── MNISTVal/nonlinear_zeroshot.pt
├── CarsVal/nonlinear_finetuned.pt
├── DTDVal/nonlinear_finetuned.pt
├── ...
├── head_CarsVal.pt
├── head_DTDVal.pt
└── ...
datasets/                       データセットごとの既存配置
```

入力パスは `config/inputs/author_fft.yaml` に定義しています。基準重み、各タスク重み、分類ヘッド、データを先に用意します。評価中の取得やヘッド再生成は行いません。`.pt` は信頼できる配布元・自分で作成したファイルだけを入力してください。

```bash
micromamba run -n tsv python prepare.py model=vit_b32 benchmark=fft8
```

`prepare.py` は重みを1個ずつ読み、元ファイルを保持して `cache/checkpoints/` に変換します。各テンソルが変換前と一致するか検査し、ヘッドの存在も確認します。変換の中断は、元重みとの照合を経て復旧します。データ分割とクラス数などの検査は、実験開始時に行います。

別の入力場所を使う場合は、準備と実験の両方で同じパスを指定します。

```bash
micromamba run -n tsv python prepare.py \
  inputs.checkpoint_root=/path/to/checkpoints inputs.data_root=/path/to/datasets
```

## 3．実験を実行する

### 固定設定

```bash
micromamba run -n tsv python run.py model=vit_b32 benchmark=fft8 method=tsvm
```

既定はα=1、FP32、GPU、バッチサイズ32です。これは係数を探索した結果ではありません。実行前に設定だけを確認できます。

```bash
micromamba run -n tsv python run.py --cfg job --resolve \
  model=vit_l14 benchmark=fft20 method=dc_merge
```

モデルは `vit_b32`・`vit_b16`・`vit_l14`、タスク集合は `fft8`・`fft14`・`fft20` から選びます。タスクと順序はそれぞれのYAMLに固定しています。共通リストの先頭N件では定義しません。

### 検証による設定選択

```bash
micromamba run -n tsv python run.py method=tsvm protocol=common_tuned \
  protocol.selection.max_validation_images=1000000
```

候補は手法YAMLの `search_candidates` にあります。比較する全手法で、入力・目的関数・画像予算・seedを合わせてください。上の画像予算は指定例です。各タスクの検証画像数と候補数に合わせて設定します。

選択指標は、タスク別正規化精度の単純平均です。同率なら宣言順で先の候補を採用します。検証済みの最良重みを保持し、その同じ重みでテストします。テスト結果は選択処理へ渡しません。評価時はタスク・split・seedから乱数を固定し、Python・NumPy・PyTorchの標準乱数状態を退避・復元します。ワーカー数も分母の識別条件に含めます。同一のvalidation/testや、識別できる物理サンプルの重複は選択開始前に拒否します。

Consensusは固定設定でもタスク別マスクの検証選択を行います。マスク選択も画像予算へ含めます。SERAの自動探索は次のように指定します。

```bash
micromamba run -n tsv python run.py method=sera protocol=sera_bayes \
  protocol.selection.max_validation_images=1000000
```

SERAは著者コード互換とは表記しません。未記載部分の局所仕様は `src/methods/sera.py` と `src/selection/sera.py` に固定しています。`paper_reference` は論文固有条件を明示して記録する区分です。名前を指定しただけで厳密再現にはなりません。

### ViT-B/32・8タスク・10手法の共通比較

今回の実験設定は `benchmark=fft8_all_images` です。Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC45、SVHN、SUN397の順で、現在のFull Fine-tuning重み・分類ヘッドを全手法で共用します。新たな学習は行いません。

SUN397には、ローカルの書き出しに含まれるGIFも使います。専用factory `sun397_all_images` を既存のadapter登録表へ追加しています。GIFは先頭フレームをRGBで読み、元画像の変更・再圧縮はしません。クラスのソート順と表示名は旧ローダーと同じです。旧 `benchmark=fft8` はGIFを除外する従来の挙動を維持します。

| タスク | 検証画像数 | テスト画像数 |
|---|---:|---:|
| Cars | 814 | 8,041 |
| DTD | 376 | 1,880 |
| EuroSAT | 2,700 | 2,700 |
| GTSRB | 2,664 | 12,630 |
| MNIST | 5,000 | 10,000 |
| RESISC45 | 2,520 | 3,150 |
| SVHN | 5,000 | 26,032 |
| SUN397 | 1,985 | 19,850 |
| 合計 | 21,059 | 84,283 |

これは現在のローカルデータに対する数です。SUN397は `sun397/train` の19,850枚からseed=0で10%を検証用に抽出し、`sun397/val` の19,850枚をtestに使います。他の7タスクは従来の分割で、例えばRESISC45の検証はtrainから抽出した2,520枚です。著者チェックポイントの学習データと検証データの厳密な独立性・ヘッドのクラス順序の由来は未確認であり、論文条件との厳密一致とは扱いません。

| 手法 | 使用するprotocol | 選択・固定条件 |
|---|---|---|
| Zero-shot、Weight Averaging | `fft8_tuned_single` | 候補1個。平均は成分削減なし |
| Task Arithmetic | `fft8_tuned_alpha` | αを選択、成分削減なし |
| TIES | `fft8_tuned_alpha` | αを選択、保持率20% |
| Consensus TA | `fft8_tuned_alpha` | 各タスクでマスクλを0.2～0.6の0.1刻みから選び、その後αを選択。合意数の下限2 |
| TSV-M、GSE-TSV | `fft8_tuned_alpha` | αを選択、成分選択は各実装の規則 |
| Iso-CTS | `fft8_tuned_alpha` | αを選択、共通空間比率0.8 |
| DC-Merge | `fft8_tuned_alpha` | αを選択、投影空間の保持率0.1%、FFT版 |
| SERA | `fft8_tuned_sera` | エネルギー閾値とαを選択、保持率0.1%、論文準拠ローカル版 |

αは0.0～3.0を0.1刻みで探索し、検証の平均正規化精度で選択します。同率なら先に評価した候補を採用します。SERAのエネルギー範囲は0.1～0.99、初期候補は0.45・0.65・0.85、最大12候補、正規化EIの停止閾値は0.01です。各エネルギーで同じ31個のαを評価します。3種類のprotocolは同じ比較条件名・目的関数・seed=0・検証予算800万画像を持ちます。Zero-shotと平均も、この比較内では探索候補1個のgridとして記録します。

予算は各手法の上限で、消費量を揃えるものではありません。SERAの予定上限は7,833,948画像、全10手法合計は検証だけで最大12,551,164画像です。正規化分母の準備と最終テストはこの探索予算に含まず、時間・評価条件は別に記録します。αが探索上端に達しても自動で範囲を拡張しません。

```bash
micromamba run -n tsv python prepare.py model=vit_b32 benchmark=fft8_all_images

bash scripts/run_fft8_b32_common_tuned_v1.sh
```

実験名 `fft8_b32_common_tuned_v1` を、実行スクリプト・結果データ・図表の共通名として使います。`fft8` はFull Fine-tuning重みによる8タスク、`b32` はViT-B/32、`common_tuned` は共通条件で検証選択する比較、`v1` は実験条件の版です。

```text
scripts/run_fft8_b32_common_tuned_v1.sh         実行手順（内部でrun.pyを呼ぶ）
results/experiments/fft8_b32_common_tuned_v1/   実験結果の元データ
└── <run_id>/                                手法ごとのresult.json・履歴・選択重み
results/reports/fft8_b32_common_tuned_v1/       上記データから作る比較表・図
├── all_methods/                             全10手法
│   ├── comparison.csv                       元の桁数を保った集計
│   ├── figures/                             平均・タスク別の精度／正規化精度
│   └── tables/                              手法×タスク・平均の数値表
└── selected_methods/                        TSV-M・Iso-CTS・DC-Merge・GSE-TSV・SERAの同形式出力
```

既存の10手法の実験は、当時READMEに記載したコマンドから `run.py` を実行したものです。その実行手順を今回同名スクリプトへ移しました。スクリプトは各手法を別プロセスで実行し、途中で失敗したら停止します。図表フォルダは改善済みの版へ統一し、`_readable` の接尾辞は外しました。

途中で失敗したら原因を確認し、`run.py` に同じ条件と該当するprotocol、`resume_from=<中断run>` を指定します。上のスクリプトを再実行すると、完了済み手法も新しい実験として再実行されます。実行中は共通コード・設定・入力を変更しないでください。

実行例は画像読み込みを4ワーカーで並列化します。`0`では実験本体が画像の読み込み・サイズ調整なども順番に行い、`4`ではCPU上の4つの補助プロセスが画像を並行して準備します。目的はGPUに画像が届くまでの待ち時間を減らすことで、GPUは1台のまま、統合式や係数候補も同じです。ローカルのSUN397検証画像256枚で、0ワーカーと前処理テンソル・ラベルが完全一致することを確認しています。ワーカー数も分母の識別条件なので、変更時は全手法で条件を揃えます。OSのプロセス間通信と共有メモリが必要です。

10手法の完了を確認してから比較します。以下のワイルドカードは、その親ディレクトリに今回の完了10件だけがある場合に使います。追加実行がある場合は比較するrunを明示してください。

```bash
micromamba run -n tsv python compare.py \
  results/experiments/fft8_b32_common_tuned_v1/*/result.json \
  --methods weight_average task_arithmetic ties consensus_ta tsvm iso_cts dc_merge sera gse_tsv zeroshot \
  --output results/reports/fft8_b32_common_tuned_v1/all_methods/comparison.csv \
  --plots results/reports/fft8_b32_common_tuned_v1/all_methods/figures \
  --task-metric both \
  --tables results/reports/fft8_b32_common_tuned_v1/all_methods/tables
```

この図表は作成済みです。再生成する場合は新しい出力先を指定します。5手法の抜粋コマンドは次節にあります。

GIFの追加でSUN397の対象と分割が変わるため、旧結果や旧分母JSONは今回の比較へ混ぜません。分母は同じ条件のキャッシュだけ再利用し、必要なものは共通評価器で測定します。平均精度はタスクを等しく重み付けし、正規化精度はタスク別FT精度との比を取ってから平均します。時間・メモリは保存結果の`performance`にあり、キャッシュや中断・再開による計測範囲の違いに注意してください。

### 81条件を列挙する

Hydraの `-m` は、既定で順次実行します。固定設定と調整設定は別の実験群として扱います。

```bash
micromamba run -n tsv python prepare.py -m \
  model=vit_b32,vit_b16,vit_l14 benchmark=fft8,fft14,fft20

micromamba run -n tsv python run.py -m \
  model=vit_b32,vit_b16,vit_l14 benchmark=fft8,fft14,fft20 \
  method=weight_average,task_arithmetic,ties,consensus_ta,tsvm,iso_cts,dc_merge,sera,gse_tsv
```

入力が存在しない条件は開始前に失敗します。他モデルや他タスクの重みを代用しません。

## 4．結果・比較表・図

各実行は、上書きしない新しいディレクトリに保存します。

```text
results/experiments/<run_id>/
├── progress.json          段階・設定・入力の識別情報
├── trials.jsonl           全候補・マスク選択・探索の履歴
├── selected.safetensors   選択時に実際に評価した完成済み重み
└── result.json            完了した結果、診断、時間・メモリ
```

配列を含む観測値がある場合は、追加で `trace.safetensors` を保存します。重みを保存しない指定でも識別情報は残します。正規化精度は、タスク別重み・分割・ヘッド・前処理・評価実装に結びつく分母を使い、比率を計算してから平均します。100%超の値も切り詰めません。

前処理は変換の順序・設定・関数の実装・関連ライブラリの版から識別します。関数のメモリアドレスを含む `repr` は使わないため、別プロセスでも同じ条件なら分母を再利用し、結果を比較できます。識別できない独自の前処理は明示的に拒否します。

分母の識別条件には、GPUの種類・計算能力、PyTorchのビルド、cuDNNの版と決定性設定なども含めます。評価値は正解数・画像数・タスク別精度・平均の整合を確認してから保存します。異常を検出した評価値を、再開時に再利用しません。

```bash
micromamba run -n tsv python compare.py \
  results/experiments/<run_a> results/experiments/<run_b> \
  --output results/reports/comparison_01/comparison.csv \
  --plots results/reports/comparison_01/figures \
  --task-metric both \
  --tables results/reports/comparison_01/tables
```

CSVにはモデル、タスク集合、手法・実装版、平均・タスク別精度、選択設定、条件グループを出します。図と表も保存結果だけから生成します。再マージや再評価は行いません。既存CSV・図・表のディレクトリへの上書きは拒否します。

`--plots` は平均精度・平均正規化精度の棒グラフと、タスク別グラフをPNG・SVGで出力します。棒の上に百分率を小数点以下2桁で表示し、横の目盛り線を描きます。タスク別の凡例はグラフの右側に置きます。正規化精度の100%は単一タスクFTモデルと同じ精度を表す基準で、上限ではありません。100の目盛りを表示し、100%超の値も描画します。

タスク別グラフは、既定では通常の精度（`Accuracy`）です。`--task-metric normalized_accuracy` で正規化精度（`Normalized Accuracy`）、`--task-metric both` で両方を出せます。保存された値を使い分け、通常の精度のラベルだけを正規化精度へ変えることはしません。

`--tables` は「行＝手法、列＝宣言された順序のタスク＋Mean」の表を、通常の精度・正規化精度それぞれについてCSV・Markdown・PNG・SVGで出力します。表の数値は百分率・小数点以下2桁です。Meanは丸める前のタスク別精度の単純平均で、画像数による重み付けはしません。元の精度の桁数は `comparison.csv` と各 `result.json` に残ります。

一部の手法だけを取り出すには、`--methods` に手法IDを表示したい順序で指定します。未指定なら全入力を使い、未知のIDや重複指定は拒否します。今回の8タスク実験から5手法を抜粋する例：

```bash
micromamba run -n tsv python compare.py \
  results/experiments/fft8_b32_common_tuned_v1/*/result.json \
  --methods tsvm iso_cts dc_merge gse_tsv sera \
  --output results/reports/fft8_b32_common_tuned_v1/selected_methods/comparison.csv \
  --plots results/reports/fft8_b32_common_tuned_v1/selected_methods/figures \
  --task-metric both \
  --tables results/reports/fft8_b32_common_tuned_v1/selected_methods/tables
```

このワイルドカードは完了した今回の10件だけが含まれる場合に使用します。同じ手法の別実装・別設定が複数含まれる場合は、図と表のラベルに実装版・設定も表示します。通常の集計CSVには常に実装版・設定を保存します。

共通の入力・数値処理・評価コードと数値環境の識別情報も比較します。手法だけを追加した場合は、共通基盤が同じなら比較できます。全体ソースの識別情報は別に残します。条件が異なる結果は通常は比較を拒否します。意図的に一覧へ出す場合は `--allow-mixed` を指定し、図は条件グループごとに分かれます。過去JSONは `--legacy` と明示的なラベルを指定した場合だけ読めます。由来不明の結果を新実験と同条件として扱いません。

```bash
micromamba run -n tsv python compare.py old_a.json old_b.json \
  --legacy --legacy-labels TSV-M AdaptiveTSV --allow-mixed \
  --output results/reports/historical_01/comparison.csv
```

観測済み診断だけを取り出す場合：

```bash
micromamba run -n tsv python scripts/collect_five_method_merge_diagnostics.py \
  --runs results/experiments/<run_a> results/experiments/<run_b> \
  --output results/reports/diagnostics_01.json
```

このスクリプトも再マージしません。取得していないSTI等は未取得として扱います。診断は `runtime.trace=none`・`summary`・`full` で指定します。fullで追加分析をする手法と、ランク等の記録のみの手法があります。取得内容は保存されたtraceが根拠です。GPUの最大使用量は分母準備の後から最終保存の前までを計測し、常駐モデルと診断の使用量を含みます。RAMの値はLinuxプロセス全期間の最大RSSです。

### TSV-MとGSE-TSVのランク配分・精度の分析

同じ実験名の保存済み結果から、各層・各タスクのランク、エネルギー、90%エネルギー到達ランク、保持率と精度差を集計します。新しいマージ・学習・推論は行いません。

```bash
micromamba run -n tsv python scripts/analyze_fft8_b32_common_tuned_v1.py
```

入力は `results/experiments/fft8_b32_common_tuned_v1/*/result.json`、出力は `results/reports/fft8_b32_common_tuned_v1/tsvm_gse_rank_accuracy/` です。`--runs <TSV-Mのrun> <GSE-TSVのrun>` で対象を明示できます。同じ手法の実行が複数あれば自動選択せずに拒否します。既存出力は上書きせず、再生成には `--output <新しい保存先>` を使います。

`layer_task_metrics.csv`、`layer_summary.csv`、`task_summary.csv` に全記録行列の値と集計を出します。主な図と相関は画像側50行列を対象にし、未使用のゼロ差分テキスト行列2個も含む52行列の集計は `scope=all_recorded_matrices` として区別します。ゼロエネルギーのr90・保持率、分母が0の増減率は空欄です。

全層の保持エネルギー率は「保持エネルギーの合計÷総エネルギーの合計」を主指標にし、層別比率の単純平均も別の列へ出します。`correlations.csv` はタスクを1標本としたPearson・Spearman相関、`layer_correlations.csv` は各行列におけるタスク間相関です。層数を標本数に数えた相関は計算しません。

`figures/` にタスク別のランクと精度差、ランク増減率のヒートマップ、散布図をPNG・SVGで保存します。代表スペクトルは精度差を見て選ばず、既定でブロック0・5・11のAttention QKV行列を使い、その曲線データを `spectral_curves.csv` に出します。`analysis.md` に定義・入力の識別情報・結果と解釈の限界を記録します。

## 5．中断した実験を再開する

```bash
micromamba run -n tsv python run.py method=tsvm protocol=common_tuned \
  protocol.selection.max_validation_images=1000000 \
  resume_from=results/experiments/<unfinished_run>
```

元と同じ設定・ソース・入力・評価条件が必要です。進捗・検証履歴・保存済み診断配列・新しい完了JSONをチェックサムで確認します。履歴が欠落した状態を新しい探索として再開しません。旧版でチェックサムを記録していない完了結果は `not_recorded` として読み、元のファイルを更新しません。完了済み実験は再開・上書きできません。中断した検証も予約した画像数を消費し、探索履歴や最良重みが改変されていれば拒否します。

## 6．単一タスク評価・ランク実験・学習

旧入口はルートに保持していますが、推論と保存は共通基盤を通ります。先に同じ入力に対して `prepare.py` を実行してください。

```bash
micromamba run -n tsv python eval_single_task_8.py \
  --model ViT-B-32 --finetuning-mode standard --device cuda

micromamba run -n tsv python eval_single_task.py \
  --model ViT-B-32 --eval-datasets MNIST,DTD --device cuda

micromamba run -n tsv python eval_single_task_for_rank.py \
  --model ViT-B-32 --eval-datasets MNIST --rank-fractions 0.1,0.5 --device cuda
```

`eval_single_task.py` の既定タスク集合は20タスクです。8タスク版は8タスクを指定します。旧入口の既定は、以前の計算順序である `base + (FT - base)` による単一タスク評価です。`--direct-ft` を付けるとFT重みそのものを評価します。どちらを使ったか結果に残します。共通比較の正規化分母はFT重みそのものの精度です。過去の共有分母JSONは自動流用・上書きしません。

ランク実験は候補を事前に固定した単一タスク研究です。各タスク・各ランクを独立した実行として保存し、9手法の共通モデル比較とは区別します。`finetune.py` の学習式は保持し、評価との接続だけを共通化しています。学習の再実験は今回の検証対象外です。

旧Hydraコマンドも利用できます。

```bash
micromamba run -n tsv python main.py model=ViT-B-32 num_tasks=8 \
  method=TSVM wandb.mode=disabled save_subfolder=my_new_comparison
```

旧設定は `config/legacy/` で読み、共通runnerへ変換します。`save_subfolder` は `results/<指定値>/` を保存先の親とし、その下に新しい実行ディレクトリを作ります。`cache_dir` も引き継ぎます。W&B設定は新runnerでは使用しません。旧係数候補は維持しますが、分母・保存形式は新基盤の条件です。過去のJSONと同じファイルを再生成するコマンドではありません。

## 7．コードの構成とデータの流れ

```text
run.py / prepare.py / compare.py       実行・準備・集計の入口
config/                               benchmark/model/inputs/method/protocol/runtime
src/core/                             共通型、重みの読み込み、数値処理
src/methods/                          各手法。完成済みの重みを返す
src/methods/experimental/             継続利用する研究手法を個別に実装
src/selection/                       検証による固定・候補一覧・専用探索
src/evaluation/                      共通推論・正規化・単一タスク評価
src/experiments/                     入力解決、登録、実行、保存、集計、作図
src/models/adapter.py                モデル・ヘッド・重み適用
src/datasets/adapter.py              明示的なtask/splitからデータを取得
src/datasets/splits.py               一つの共通分割計算
src/compat/                          旧CLI・旧数値API・旧結果・学習との接続
src/utils/                           学習用補助と必要な互換export
scripts/                             現在利用するデータ準備・調査の入口
archive/                             過去の成果物と旧コード
```

主なデータは次のとおりです。

| 変数・型 | 内容と渡し先 |
|---|---|
| `spec: ExperimentSpec` | 解決済みの型付き設定。入力解決・runnerへ渡す |
| `inputs: MergeInputs` | 基準重み・タスク重みの読取元と順序。手法へ渡す |
| `context: MergeContext` | デバイス・精度・一時キャッシュ。画像・ラベルは含まない |
| `output: MergeOutput` | α適用まで済んだCPUの `weights` と観測した `trace` |
| `selected: SelectionResult` | 選択設定と検証済みの `output`。そのままテストへ渡す |
| `metrics: EvaluationResult` | タスク別正解数・画像数・精度と、タスクの単純平均 |

数値関数はHydra・評価器・W&B・作図を呼びません。旧runnerへ戻る分岐もありません。SVD系は層単位で処理し、TIESは全体の閾値・符号を維持してブロック化します。αだけが変わる場合はマージ差分を再利用します。

## 8．新しい手法を追加する

基本的に変更する場所は4か所です。

1. `src/methods/my_method.py`：型付きパラメータと `merge()`。
2. `config/method/my_method.yaml`：ID・パラメータ・候補。
3. `src/experiments/registry.py`：計算、選択、実装版、出典、必要なら入力検査。
4. `tests/`：参照実装・境界条件・入力非破壊・診断の有無を検証。

```python
# インターフェースの説明例
from dataclasses import dataclass
from src.core.types import MergeOutput

@dataclass(frozen=True)
class MyParams:
    alpha: float = 1.0


def merge(inputs, params, context, fitted_state=None):
    weights = ...  # 手法の計算とalpha適用までを完了させる
    return MergeOutput(weights=weights, trace={})
```

固定・候補一覧からの選択は `selection/search.py` を再利用します。専用探索が必要な場合だけ選択関数を追加し、同じ検証画像予算と履歴管理を使います。入力条件の制約は登録時の `validate_inputs(inputs, params)` で開始前に検査できます。通常、評価器・保存処理・作図へ手法名の分岐を追加する必要はありません。

数値規則を変更した場合は実装版を変更します。参照テスト側を書き換えて一致させてはいけません。研究用設定は `method=experimental/adaptive_tsv` などで指定します。

## 9．データセット・タスク集合を追加する

新しいタスク集合は `config/benchmark/my_tasks.yaml` に明示します。既存の8・14・20タスク設定を変更して別の集合にしないでください。

```yaml
name: my_tasks
task_ids: [MyTask]
tasks:
  MyTask:
    checkpoint: "{model}/MyTask/nonlinear_finetuned.pt"
    head: "{model}/head_MyTask.pt"
    class_names: [class_a, class_b]
    validation:
      adapter: imagefolder
      path: my_task/validation
    test:
      adapter: imagefolder
      path: my_task/test
```

重み・ヘッドは `inputs.checkpoint_root`、画像は `inputs.data_root` が基準です。ImageFolderのクラスディレクトリのソート順がラベル番号になります。validation/test/ヘッドで同じ順序を使います。

```bash
micromamba run -n tsv python prepare.py benchmark=my_tasks
micromamba run -n tsv python run.py benchmark=my_tasks method=tsvm
```

汎用adapterは `legacy`・`imagefolder`・`tensor_file`・`hf_disk` で、SUN397の全画像用には `sun397_all_images` があります。新しい保存形式や読み込み規則が必要なときは `src/datasets/adapter.py` の `DATASET_ADAPTERS` にfactoryを登録します。factoryは `(split_spec, data_root, preprocess, runtime)` を受け、`(dataset, ordered_class_names)` を返します。取得処理は含めません。

`tensor_file` は前処理済みの `images`・整数 `labels`・ラベル番号順の `class_names` を持つ `.pt`、`hf_disk` は取得済みの `save_to_disk` 形式です。画像・ラベル・分割・クラス順序・実装の変更を識別できる情報が必要です。独自Datasetでは、異なるsplit間の部分重複も検査するため、元データ源を含む安定した `sample_ids` を行順に公開してください。独自の内部乱数生成器を使う場合も、同じ条件で同じ画像を返す責任はfactory側にあります。追加時には、ヘッドとの不一致検出とキャッシュ失効もテストしてください。

`hf_disk.path` は分割ごとに保存した単一Datasetを指定します。行の選択・順序、画像の内容、ラベルを識別し、外部画像への参照があれば画像ファイルの内容も確認します。外部画像の物理パスによる重複は検査できますが、埋め込み画像だけでは別々の書き出し間の元サンプルIDは分かりません。物理IDの確認範囲は `membership_status`（complete／partial／unavailable）へ記録します。全画像の元サンプルIDを検査する場合は、それを公開する独自factoryを用意してください。

分類ヘッドの `normalize` は `.pt` では真偽値、safetensorsのメタデータでは文字列 `true`／`false` を使います。`weighted`・`reconstructed`・`runtime.save_weights` も真偽値を指定します。文字列 `"false"` を真として扱うような型違いや、綴り間違いは入力エラーとして報告します。診断テンソルのキーには `/` や `~` を含めても、階層の異なるキーと衝突しない形で保存します。

## 10．過去の成果物の場所

以下の5フォルダは、中身を保って `archive/` へ移動しました。

```text
archive/
├── five_method_figures/       過去の図・SVG・元CSV
├── presentation_figures/      過去の発表用図・生成スクリプト・説明
├── repro_metadata/            当時の環境・変更記録
├── reproduction_metadata/    別に記録された環境・重みのチェックサム
├── run_logs/                  過去の実行ログ
└── legacy_code/               置き換え前のソース・旧特殊実験
```

2種類のメタデータは別の記録なので統合していません。アーカイブ内のスクリプトは当時のソースとして保持しており、現在の入口から自動実行しません。記録中の当時のパスもそのままです。新しい図は `results/reports/`、新しい実験は `results/experiments/` に保存します。既存の `results/` 内の研究結果、データ、モデル、`logs/` は移動・上書きしていません。

## 11．テストと確認範囲

```bash
micromamba run -n tsv python -B -m unittest discover -s tests -v

# ローカルの著者重みを使う追加確認
TSV_REAL_CHECKPOINT_TESTS=1 micromamba run -n tsv python -B -m unittest discover -s tests -v

# 全パラメータと8ヘッドの固定入力比較（GPU・準備済みB/32重みが必要）
TSV_FULL_MODEL_TESTS=1 micromamba run -n tsv python -B -m unittest discover -s tests -p 'test_full_model.py' -v

# GPU閾値境界・矩形行列と公式参照の比較
TSV_GPU_AUDIT_TESTS=1 micromamba run -n tsv python -B -m unittest discover -s tests -p 'test_gpu_boundaries.py' -v
```

通常のテストはCPU・小さな人工入力で動きます。複数ワーカーの検査にはOSのプロセス間通信が必要です。テストはデータや重みをダウンロードしません。固定参照は `tests/reference/`、移行前の研究用実装は `archive/legacy_code/` に保持しています。テストは今後の変更を検出するため、リファクタリング後も残します。

この移行では、ViT-B/32・8タスクの保護対象4手法について、全158テンソルと8ヘッドの固定入力logits・予測が参照計算と完全一致することをGPUで確認しました。この比較は変更前後の重みを同じ評価adapterへ適用する検査です。OpenAI側のモデル構築との一致は別に、通常テストの小型モデルと、キャッシュ済みOpenAI重みを使う実モデルで確認します。実モデル検査は取得済みの `ViT-B-32.pt` が必要で、別配置では `TSV_OPENAI_CHECKPOINT` を指定できます。テスト中のダウンロードは行いません。過去のMNIST実データによる2候補の選択・テスト・保存はv2での動作確認であり、v3の実精度の根拠には使いません。ViT-B/16・ViT-L/14の配布重みはローカルに存在せず、それらの実精度評価は未実施です。

WSLで `libcuda.so` の探索に失敗する場合は、ドライバを確認したうえで、実行コマンドに `LD_LIBRARY_PATH=/usr/lib/wsl/lib` を指定します。このOSの標準ドライバ配置を確認してから使ってください。数値設定やデバイスを自動変更する処理はありません。

実データを使うGPU精度比較と、単体テスト・固定入力の一致検査は別の確認です。異なるPyTorch/GPU間のbitwise一致を保証するものではありません。正規化分母やヘッドのクラス順序の由来が未確認なら、その状態を結果に記録します。

| 問題 | 対応 |
|---|---|
| `Not prepared` | 同じ入力・モデル・タスク集合で `prepare.py` を実行 |
| ヘッド／データ不足 | 評価前に必要なローカル入力を用意 |
| CUDAを利用できない | 実行環境を確認し、意図した場合だけCPUを明示 |
| 検証予算不足 | 履歴を保持し、予算を設定した別実行を開始 |
| 比較条件不一致 | 入力・分割・評価条件を揃える。意図した一覧のみ `--allow-mixed` |
| 再開時の識別不一致 | 過去実行を改変せず、変更後の条件で新しい実行を開始 |
