# Graduation Research: Deep Learning Based Text Mining on PBL Data

本リポジトリは、Project-Based Learning (PBL) における学習者の記述データを対象に、Attention付きニューラルネットワークとクラスタリング可視化を用いて**思考構造を分析・可視化**する研究用コードです。

---

## 🔧 ディレクトリ構成

```yaml
├── .gitignore
├── README.md
├── requirements.txt
└── src
    ├── bert_model_loader.py
    ├── data_reader.py
    ├── dual_attention.py
    ├── executer.py
    ├── sentence_encoder.py
    └── sentence_group.py
```
---
## 🚀 実行方法
### 環境構築
```bash
#仮想環境の作成
python3 -m venv .venv-pbl-nlp
source .venv-pbl-nlp/bin/activate
#パッケージインストール
pip install -r requirements.txt
```
### 実行
```bash
python src/executer.py
```

### 実行結果
![](./outputs/graphs/combined_bigraph.png)