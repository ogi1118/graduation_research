# Graduation Research: Deep Learning Based Text Mining on PBL Data

本リポジトリは、Project-Based Learning (PBL) における学習者の記述データを対象に、Attention付きニューラルネットワークとクラスタリング可視化を用いて**思考構造を分析・可視化**する研究用コードです。

---

## 🔧 ディレクトリ構成

```yaml
.
├── deep/ 
├── simplified_deep/ 
├── dataset20240126.csv # 入力データ
├── others
└── requirements.txt # 使用パッケージ一覧
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
python simplified_deep/executer.py
```