import pandas as pd
import os


def convert_csv_to_tsv(input_path, output_path, encoding='utf-8', sep_in=None):
    # 区切り文字の自動判定（Noneならカンマ、セミコロン、タブを試す）
    if sep_in is None:
        with open(input_path, 'r', encoding=encoding) as f:
            sample = f.readline()
            if '\t' in sample:
                sep_in = '\t'
            elif ';' in sample:
                sep_in = ';'
            else:
                sep_in = ','
    df = pd.read_csv(input_path, sep=sep_in, encoding=encoding)
    df.to_csv(output_path, sep='\t', index=False,
              encoding=encoding, lineterminator='\n')


if __name__ == "__main__":
    # 変換対象ファイル
    files = [
        ("../GPBL_0901.csv", "../data/GPBL_0901.tsv"),
        ("../GPBL_origin.csv", "../data/GPBL_origin.tsv")
    ]
    for in_path, out_path in files:
        print(f"Converting {in_path} -> {out_path}")
        convert_csv_to_tsv(in_path, out_path)
    print("変換完了: /data ディレクトリに保存しました")
