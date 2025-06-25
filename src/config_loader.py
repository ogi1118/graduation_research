import os
import json

# ──────────── 設定ファイル読み込み ────────────
# このファイル（config_loader.py）の２階層上をプロジェクトルートとみなす
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
# プロジェクトルート直下の config.json を読み込む
CONF_PATH = os.path.join(PROJECT_ROOT, "config.json")
with open(CONF_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# ──────────── 相対パス→絶対パス変換 ────────────
config = {}
for key, val in raw.items():
    # 文字列かつ絶対パスでないものはプロジェクトルート基準で絶対パスに
    if isinstance(val, str) and not os.path.isabs(val):
        config[key] = os.path.abspath(os.path.join(PROJECT_ROOT, val))
    else:
        config[key] = val

if __name__ == "__main__":
    # 設定内容を表示
    print(json.dumps(config, indent=2, ensure_ascii=False))
