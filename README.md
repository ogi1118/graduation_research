# Graduation Research: Deep Learning Based Text Mining on PBL Data

本リポジトリは、Project-Based Learning (PBL) における学習者の記述データを対象に、Attention付きニューラルネットワークとクラスタリング可視化を用いて**思考構造を分析・可視化**する研究用コードです。

---

## 📋 目次
- [ディレクトリ構成と各スクリプトの役割](#ディレクトリ構成と各スクリプトの役割)
- [実行フロー](#実行フロー)
- [環境構築と実行方法](#環境構築と実行方法)
- [設定](#設定)

---

## 🔧 ディレクトリ構成と各スクリプトの役割

### `src/` ディレクトリ：メイン処理

#### **config_loader.py**
- **役割**: JSON形式の設定ファイル(`config.json`)を読み込み、相対パスを絶対パスに変換する
- **出力**: グローバル変数`config`を提供（他のスクリプトから import して使用）
- **内容**:
  - ファイルパス、出力ディレクトリ、外れ値検出パラメータなど

#### **data_reader.py**
- **役割**: TSVファイルから学習者の記述データを読み込み、文単位で分割・処理する
- **主要クラス**:
  - `SentenceFinder`: spaCyを使って自然言語処理（文の分割）を行う。コンセントフィルターを適用可能
  - `SentenceConverter`: 複数カラム間の文ペアを準備
- **処理フロー**:
  1. CSVファイルをpandasで読み込み
  2. spacy (`en_core_web_sm`) で文単位に分割
  3. 最大文数を記録（パディング用）
  4. オプション: コンセント情報でフィルタリング

#### **bert_model_loader.py**
- **役割**: HuggingFaceから事前学習済みBERTモデルをロードし、テキストをベクトル化する
- **使用モデル**: `sentence-transformers/all-distilroberta-v1`
- **主要メソッド**:
  - `get_encodings()`: テキストをトークン化し、アテンション可能なテンソルに
  - `get_CLS()`: [CLS] トークンを抽出してセンテンス埋め込みを取得
  - `get_avepool()`: トークン埋め込みを平均プーリング

#### **sentence_encoder.py**
- **役割**: BERTモデルを使ってテキストを埋め込みベクトルに変換（バッチ処理対応）
- **主要クラス**:
  - `SentenceEncoder`: BertLoaderをラップ
    - `encode()`: 単一テキストのベクトル化
    - `doc_encode()`: 複数ドキュメント・複数文のバッチ埋め込み
    - `encode_sentences()`: SentenceGroupListをテンソルに変換

#### **dual_attention.py**
- **役割**: 2つの入出力テキスト列間の対応関係をニューラルネットワークで学習（Attention機構を搭載）
- **主要クラス**:
  - `Attention`: マスク付きのソフトマックスアテンション層
  - `AttentionEncoder`: テキスト埋め込みに対してアテンション重みを計算し、圧縮埋め込みを生成
  - `Associator`: 入力側の埋め込みから出力側を予測するニューラルネット
  - `DualAttention`: 全処理の統合クラス
    - `read()`: data_readerでテンソル化
    - `train()`: 学習ループ（MSEロスで最適化）
    - `get_attention()`: 学習済みAttentionの重みを取得

#### **executer.py**
- **役割**: メイン実行スクリプト。全体の処理フローを制御
- **処理フロー**:
  1. config.jsonから設定を読み込み
  2. 隣接する2つのカラム対（"Motivation for the Actions" → "Actions Taken Today" など）に対して以下を繰り返す：
     - `Executer.train()`: DualAttentionで学習
     - `AttentionEvaluator.evaluate()`: アテンション値をCSVに出力
     - `Visualizer.save_cluster_info()`: クラスタリング結果を保存
  3. `CombinedVisualizer`: 全カラムペアの結果を統合し、可視化グラフを生成

#### **visualizer.py**
- **役割**: 埋め込みベクトルとアテンション結果を可視化・クラスタリング
- **主要機能**:
  - `Visualizer`:
    - 埋め込みベクトル・アテンション値・マスクをNumpy形式で `outputs/embed/` に保存
    - UMAP + DBSCANでクラスタリング（2D散布図生成）
  - `CombinedVisualizer`:
    - 複数カラムペアのクラスタ情報を統合
    - ネットワークグラフ、トピックグラフを生成
    - KeyBERTでキーワード自動抽出

#### **combined_visualizer_extensions.py**
- **役割**: `Visualizer` の拡張機能。外れ値検出とグラフ生成
- **主要関数**:
  - `add_outlier_info()`: パーセンタイル法またはLOF（Local Outlier Factor）で外れ値を検出

#### **outlier_detector.py**
- **役割**: クラスタ内の外れ値を検出
- **方法**:
  - パーセンタイル法: コサイン類似度で高い乖離度のサンプルを外れ値と判定
  - LOF法: 局所的密度に基づく異常検出

#### **day_clusterer.py**
- **役割**: タイムスタンプ情報でデータを日単位に分割し、各日のクラスタリング結果を生成
- **用途**: 時間軸に沿った学習思考の変化を追跡

#### **timeline_merger.py**
- **役割**: 複数日のクラスタ情報を時系列で統合し、クラスタの遷移を追跡
- **機能**: クラスタIDの対応付け、トピックの時間的進化を可視化

#### **sentence_group.py**
- **役割**: データ構造定義
- **クラス**:
  - `SentenceGroup`: 複数カラムの文を辞書形式で管理
  - `SentenceGroupList`: SentenceGroupのリスト
  - `TensorGroup`: テンソルバージョン（埋め込み用）

---

### `embedding_experiment/` ディレクトリ：埋め込み実験

#### **embed.py**
- **役割**: SentenceTransformersモデルのロード・キャッシュ、テキスト埋め込み、コサイン類似度計算
- **用途**: 他のスクリプトでモデルを統一

#### **analysis_choices.py**
- **役割**: 選択肢（複数の文）のベクトル空間での相互類似度を可視化
- **出力**: CSV、ヒートマップ画像

#### **analysis_responses.py**
- **役割**: 自由記述回答を選択肢埋め込みと比較し、スコア推定・精度評価
- **出力**: 推定スコアのCSV、評価指標

#### **classify_example.py**
- **役割**: テキスト分類の例示スクリプト

#### **compare_\*.py** シリーズ
- **役割**: 異なる埋め込みモデルやデータセット間の類似度を比較分析

---

### `tmpSrc/` ディレクトリ

#### **csv_converter.py**
- **役割**: 一時的な補助スクリプト（データ形式変換など）

---

## 🔄 実行フロー

### 現在の環境（config.json が指定する場合）での動き

```
1. [初期化]
   ├─ config.json を読み込み
   │  └─ ファイルパス、カラム名、パラメータを取得
   └─ 出力ディレクトリ (outputs/, tmp/) を作成

2. [データ準備]
   ├─ data_reader.py: GPBL_0901.tsv を読み込み
   │  ├─ spacy で文単位に分割
   │  ├─ コンセント情報でフィルタリング
   │  └─ 最大文数を記録
   └─ sentence_encoder.py: テキストを BERT で埋め込み

3. [カラムペア処理]（隣接2カラムペア）
   └─ for i in [0, 1]: ("Motivation for Actions" → "Actions Taken Today", "Actions Taken Today" → "Actions Planned for Tomorrow")
       ├─ [モデル学習]
       │  ├─ DualAttention.read(): テンソル化（形状: [batch, max_sent, embedding_dim]）
       │  ├─ DualAttention.train(): epochs=1000 で学習
       │  │  └─ AttentionEncoder がカラム間の対応を学習
       │  └─ Attention 重みを保存
       │
       ├─ [評価・出力]
       │  ├─ AttentionEvaluator.evaluate(): アテンション値を CSV に出力
       │  │  └─ outputs/csvs/[Motivation for Actions]2[Actions Taken Today].csv
       │  └─ Visualizer: 埋め込みとアテンションを numpy 形式で保存
       │     └─ outputs/embed/{col_pair}/*.npy
       │
       └─ [クラスタリング]
          ├─ UMAP で次元削減
          ├─ DBSCAN でクラスタリング
          ├─ 散布図を生成
          └─ tmp/ に cluster_info JSON を保存

4. [クロスカラム統合]
   ├─ CombinedVisualizer.merge_and_recluster():
   │  └─ 全カラムペアの結果を統合
   ├─ 外れ値検出 (add_outlier_info):
   │  └─ パーセンタイル法または LOF で異常検出
   └─ CombinedVisualizer.draw_merged_graph():
      └─ ネットワークグラフを生成（outputs/graphs/）

5. [トピック抽出]
   └─ CombinedVisualizer.save_major_topics_graph():
      └─ KeyBERT で主要トピック抽出・可視化
```

### 出力ファイル

```
outputs/
├── csvs/
│   └─ [Motivation for Actions]2[Actions Taken Today].csv (アテンション値)
│   └─ [Actions Taken Today]2[Actions Planned for Tomorrow].csv (アテンション値)
├── embed/
│   ├─ Motivation for the Actions_Actions Taken Today/
│   │  ├─ input_embeddings.npy
│   │  ├─ output_embeddings.npy
│   │  ├─ input_attention.npy
│   │  ├─ output_attention.npy
│   │  ├─ input_mask.npy
│   │  ├─ output_mask.npy
│   │  └─ sample_ids.json
│   └─ Actions Taken Today_Actions Planned for Tomorrow/
│      └─ (同様のファイル群)
└── graphs/
    ├─ merged_graph.png (統合ネットワークグラフ)
    ├─ topics_by_column.png (トピック抽出結果)
    └─ ...

tmp/
├─ cluster_info_<col_pair>_<idx>_<jdx>.json (クラスタ情報)
└─ cluster_info_<date>_<idx>_<jdx>.json (日単位のクラスタ情報)
```

---

## 🚀 環境構築と実行方法

### 前提条件
- Python 3.8 以上
- CUDA対応GPU推奨（CPUでも動作するが遅い）

### 環境構築

```bash
# リポジトリをクローン
cd /home/al22091/graduation_research

# 仮想環境の作成
python3 -m venv .venv-pbl-nlp

# 仮想環境の有効化
source .venv-pbl-nlp/bin/activate

# パッケージのインストール
pip install -r requirements.txt

# spacy の言語モデルをダウンロード
python -m spacy download en_core_web_sm
```

### 実行

```bash
# 仮想環境を有効化
source .venv-pbl-nlp/bin/activate

# メイン処理を実行
python src/executer.py
```

実行時間の目安：
- 約1000エポック × 複数カラムペア × BERT処理 = **数分～数十分**（GPUあり）
- CPU環境の場合は**1時間以上**かかる可能性あり

---

## 🔩 設定

### config.json

```json
{
  "file_name": "/path/to/data/GPBL_0901.tsv",
  "col_names": [
    "Motivation for the Actions",
    "Actions Taken Today",
    "Actions Planned for Tomorrow"
  ],
  "outputs_dir": "outputs",
  "tmp_dir": "tmp",
  "outputs_graphs_dir": "outputs/graphs",
  "outputs_csvs_dir": "outputs/csvs",
  "embeddings_dir": "outputs/embed",
  "outlier_method": "percentile",
  "outlier_percentile": 95,
  "outlier_lof_n_neighbors": 20,
  "outlier_epsilon": 1e-6
}
```

**各パラメータの説明**:
- `file_name`: 入力TSVファイルのパス
- `col_names`: 処理対象のカラム名（隣接ペアで処理）
- `outputs_dir`: 出力ディレクトリ
- `outlier_method`: 外れ値検出方式（`"percentile"` or `"lof"`）
- `outlier_percentile`: パーセンタイル法の閾値（%）
- `outlier_lof_n_neighbors`: LOF法の近傍数

---

## 📦  主要パッケージ

| パッケージ | 用途 |
|-----------|------|
| `torch` | ニューラルネットワーク学習 |
| `transformers` | BERT モデルロード |
| `sentence-transformers` | テキスト埋め込み |
| `spacy` | 自然言語処理（文分割） |
| `umap-learn` | 次元削減 |
| `scikit-learn` | クラスタリング（DBSCAN、LOF） |
| `keybert` | キーワード自動抽出 |
| `networkx` | ネットワーク分析・可視化 |
| `matplotlib`, `seaborn` | グラフ描画 |
| `pandas`, `numpy` | データ処理 |

---

## 🎯 研究の流れ

```
学習者の記述データ (3カラム)
         ↓
    [テキスト埋め込み]
         ↓
  [Dual Attention 学習]
  ├─ カラム間の対応関係を学習
  └─ アテンション重みを記録
         ↓
    [クラスタリング]
  ├─ UMAP で次元削減
  ├─ DBSCAN でクラスタ形成
  └─ 外れ値を検出
         ↓
    [可視化・分析]
  ├─ ネットワークグラフ
  ├─ トピック抽出
  └─ 時系列追跡（オプション）
         ↓
   思考構造の可視化・解釈
```

---

## � 先行研究の実装について

本リポジトリは、**先行研究の実装に基づいて拡張・改善**されています。

### 先行研究の初期実装（2025年5月14日）

初期コミット（`31cd798 - Clean repo with correct scope`）時点での構成：

```
初期ディレクトリ構成:
├── deep/                           # メインの実装
│   ├── bert_model_loader.py
│   ├── data_reader.py
│   ├── dual_attention.py
│   ├── executer.py
│   ├── sentence_encoder.py
│   └── test.py
├── simplified_deep/                # 簡略版実装
│   ├── bert_model_loader.py
│   ├── data_reader.py
│   ├── dual_attention.py
│   ├── executer.py
│   ├── sentence_encoder.py
│   ├── sentence_group.py
│   └── test.py
└── old/                            # 旧実装（参考）
    ├── clustering.py
    ├── preprocess.py
    ├── token_element.py
    └── w2v.py
```

### 開発の流れ

| 段階 | 主要な変更 |
|------|----------|
| **初期実装** | 基本的なDual Attention、BERT埋め込み、データ読み込み |
| **2カラム→3カラム対応** | 複数カラムペアの処理にループ対応（feature/3columns-own） |
| **ディレクトリ構造整理** | `deep/` と `simplified_deep/` の統合 |
| **Visualizer追加** | クラスタリング、グラフ可視化機能を実装 |
| **設定ファイル対応** | `config.json` での動的パラメータ管理 |
| **統合グラフ生成** | `CombinedVisualizer` で複数カラムペアを統合可視化 |

### 先行研究の実装の復元

`research_implementation/prior_work/` ディレクトリに、初期コミット時点の先行研究実装を復元・保存しています。

**復元されたファイル**:
- **deep/**: メイン実装版
  - `deep_executer.py` - 2カラム対応版メイン実行スクリプト
  - `deep_dual_attention.py` - Dual Attentionの基本実装
  - `deep_data_reader.py` - データ読み込みの初期実装
  - `deep_sentence_encoder.py` - 文埋め込みの初期実装
  - `deep_bert_model_loader.py` - BERTモデルローダーの初期実装

- **simplified_deep/**: 簡略版実装
  - `simplified_deep_executer.py` - 複数カラムペア対応版（ループ処理）
  - `simplified_deep_dual_attention.py` - 改良版Dual Attention
  - `simplified_deep_data_reader.py` - 拡張版データ読み込み
  - `simplified_deep_sentence_encoder.py` - 拡張版文埋め込み
  - `simplified_deep_sentence_group.py` - 構造化データ管理（新規）
  - `simplified_deep_bert_model_loader.py` - BERTモデルローダー

- **old/**: 旧実装（参考用）
  - `old_clustering.py` - 従来のクラスタリング手法
  - `old_preprocess.py` - 前処理スクリプト
  - `old_token_element.py` - トークン処理
  - `old_w2v.py` - Word2Vec埋め込み

これらのファイルは**参考用**として保存されており、現在の実装（`src/`）がメイン処理になります。
| **外れ値検出機能** | パーセンタイル法・LOF法による異常検出 |
| **時系列対応** | `day_clusterer.py`・`timeline_merger.py` でタイムスタンプ支援 |

### 現在の実装との違い

| 項目 | 先行研究 | 現在の実装 |
|------|--------|----------|
| ディレクトリ構成 | `deep/`、`simplified_deep/`、`old/` | `src/`、`embedding_experiment/`、`tmpSrc/` |
| 設定方式 | ハードコード主体 → `config.json` | `config.json` + `config_loader.py` |
| クラスタリング | 基本的なDBSCAN | UMAP + DBSCAN + 外れ値検出 |
| 可視化 | 散布図のみ | ネットワークグラフ、トピック抽出、時系列対応 |
| データ構造 | `SentenceConverter` | `SentenceGroup`、`SentenceGroupList`、`TensorGroup` |
| モデル | sentence-transformers/all-distilroberta-v1 | 同一（2.6.1へアップグレード） |

### コミット履歴の主要マイルストーン

- **初期実装段階** (コミット: `31cd798` - `a995562`)
  - 基本的なDual Attentionアーキテクチャ
  - 2カラム間の対応学習
  
- **機能拡張段階** (コミット: `5dcd5be` - `dfddacd`)
  - 3カラム対応
  - 出力ディレクトリ管理の改善

- **可視化・分析強化** (コミット: `c75a385` 以降)
  - README整備
  - グラフ生成機能
  - ネットワーク分析

---

## �📝 ライセンス

MIT License（詳細はプロジェクトルートの LICENSE ファイルを参照）
