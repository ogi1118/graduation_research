import torch
import numpy as np
import random
import os
import pandas as pd
import tempfile
from config_loader import config
import json
from dual_attention import DualAttention
from visualizer import Visualizer, CombinedVisualizer
import matplotlib
from combined_visualizer_extensions import add_outlier_info
matplotlib.use('Qt5Agg')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


class Executer:
    def __init__(self, file_name, col_name_from, col_name_to, consent_filter=None) -> None:
        self.encoded_vector_dim = 20
        self.da = DualAttention(self.encoded_vector_dim)
        self.device = self.da.which_device()
        self.file_name, self.col_name_from, self.col_name_to = file_name, col_name_from, col_name_to
        self.consent_filter = consent_filter
        self.sentence_lists = None

    def train(self, max_sentence_number=20, epochs=100):
        input_tensor, input_padding_flags, output_tensor, output_padding_flags = self.da.read(max_sentence_number=max_sentence_number, file_name=self.file_name,
                                                                                              col_name_from=self.col_name_from, col_name_to=self.col_name_to, consent_filter=self.consent_filter)
        in_encoder, out_encoder, assoc = self.da.train(
            input_tensor, input_padding_flags, output_tensor, output_padding_flags, epochs=epochs)
        self.sentence_lists = self.da.get_sentence_lists()
        return in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags

    def get_sentence_lists(self):
        return self.sentence_lists

    def which_device(self):
        return self.device


class AttentionEvaluator:
    def __init__(self, in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists, device="cpu") -> None:
        self.encoders = [in_encoder, out_encoder]
        for encoder in self.encoders:
            encoder.eval()
        self.sentence_tensors = [input_tensor, output_tensor]
        self.sentence_lists = sentence_lists
        self.device = device

    def evaluate(self, output_file_name):
        with open(output_file_name, mode="w") as f:
            in_or_out = ["-----------in-----------",
                         "-----------out-----------"]
            attention_values_list = []
            for i, (encoder, sentence_tensor) in enumerate(zip(self.encoders, self.sentence_tensors)):
                attention_values = encoder.get_attention(sentence_tensor)
                attention_values_list.append(attention_values)
                f.write(in_or_out[i]+"\n")
                for j in range(len(self.sentence_lists)):
                    sentence_list = self.sentence_lists[j][i]
                    f.write("..."+str(j)+"................."+"\n")
                    for k in range(len(sentence_list)):
                        f.write(
                            sentence_list[k]+":"+self.t2str(attention_values[j, k])+"\n")
        print("Done.")
        return attention_values_list

    def t2str(self, t):
        # if self.device=="cpu":
        #     l = [str(val)+"," for val in t.detach().numpy()]
        # else:
        #     l = [str(val)+"," for val in t.cpu().detach().numpy()]
        # result = "".join(l)
        # result = result[:-1]
        if self.device == "cpu":
            result = str(t.detach().numpy())
        else:
            result = str(t.cpu().detach().numpy())
        return result


if __name__ == "__main__":
    # GPBL_origin.tsvファイルを使用し、同意したユーザーのデータのみをフィルタリング
    consent_filter_value = "I can provide data to the research to improve the global PBL"
    original_file_name = "/home/al22091/graduation_research/data/GPBL_origin.tsv"

    # GPBL_origin.tsvを読み込み、日付別にフィルタリング
    df_origin = pd.read_csv(original_file_name, sep="\t")
    df_origin = df_origin[df_origin["Consent to use your data to research"]
                          == consent_filter_value]

    # ユニークな日付を取得
    dates = sorted(
        [d for d in df_origin["Today's date"].unique() if pd.notna(d)])

    col_names = config["col_names"]
    outputs_dir = config["outputs_dir"]
    tmp_dir = config["tmp_dir"]
    outputs_graphs_dir = config["outputs_graphs_dir"]
    outputs_csvs_dir = config["outputs_csvs_dir"]

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(outputs_graphs_dir, exist_ok=True)
    os.makedirs(outputs_csvs_dir, exist_ok=True)

    # 日付ごとにデータを処理
    for date_str in dates:
        print(f"\n{'='*60}")
        print(f"Processing date: {date_str}")
        print(f"{'='*60}")

        # 日付でフィルタリングしたデータをテンポラリファイルに保存
        df_date = df_origin[df_origin["Today's date"] == date_str]

        # テンポラリファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as tmp_file:
            tmp_file_name = tmp_file.name
            df_date.to_csv(tmp_file_name, sep='\t', index=False)

        try:
            # 日付ごとの出力ディレクトリを作成
            date_key = date_str.replace('/', '-')
            date_outputs_graphs_dir = os.path.join(
                outputs_graphs_dir, date_key)
            date_outputs_csvs_dir = os.path.join(outputs_csvs_dir, date_key)
            date_tmp_dir = os.path.join(tmp_dir, date_key)
            date_embeddings_dir = os.path.join(
                config["embeddings_dir"], date_key)

            os.makedirs(date_outputs_graphs_dir, exist_ok=True)
            os.makedirs(date_outputs_csvs_dir, exist_ok=True)
            os.makedirs(date_tmp_dir, exist_ok=True)
            os.makedirs(date_embeddings_dir, exist_ok=True)

            # 各カラムペアを処理
            for i in range(len(col_names) - 1):
                j = i + 1
                col_name_from = col_names[i]
                col_name_to = col_names[j]
                print(f"  Processing: {col_name_from} -> {col_name_to}")

                executer = Executer(
                    tmp_file_name, col_name_from, col_name_to, consent_filter=None)
                in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(
                    max_sentence_number=20, epochs=1000)
                sentence_lists = executer.get_sentence_lists()
                ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor,
                                        output_tensor, sentence_lists, device=executer.which_device())
                output_file_name = os.path.join(
                    date_outputs_csvs_dir, f"[{col_name_from}]2[{col_name_to}].csv")
                attention_value_lists = ae.evaluate(output_file_name)

                vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags,
                                 sentence_lists, attention_value_lists, device=executer.which_device(),
                                 col_name_from=col_name_from, col_name_to=col_name_to,
                                 date_key=date_key, output_graphs_dir=date_outputs_graphs_dir,
                                 embeddings_dir=date_embeddings_dir, tmp_dir=date_tmp_dir)
                vis.save_cluster_info(eps=0.7, min_samples=2)

            # 必要なファイルが揃っているか確認
            missing_files = []
            for i in range(len(col_names) - 1):
                j = i + 1
                col_name_from = col_names[i]
                col_name_to = col_names[j]
                stage_dir = os.path.join(
                    date_embeddings_dir, f"{col_name_from}_{col_name_to}")
                for fname in ["input_embeddings.npy", "output_embeddings.npy", "input_attention.npy", "output_attention.npy", "input_mask.npy", "output_mask.npy"]:
                    fpath = os.path.join(stage_dir, fname)
                    if not os.path.exists(fpath):
                        missing_files.append(fpath)
            if missing_files:
                print("  ✗ Skipping merged graphs because required files are missing:")
                for mf in missing_files:
                    print(f"    - {mf}")
                continue

            # 日付ごとのマージグラフとメジャートピックスを生成
            try:
                print("  Generating merged graphs...")
                merged_infos = CombinedVisualizer.merge_and_recluster(
                    eps=0.7, min_samples=3, date_key=date_key, tmp_dir=date_tmp_dir,
                    embeddings_dir=date_embeddings_dir)
                merged_infos = add_outlier_info(
                    merged_infos, method="percentile", percentile=90, lof_n_neighbors=2, epsilon=1e-6)

                CombinedVisualizer.draw_merged_graph(
                    merged_infos, date_key=date_key, output_graphs_dir=date_outputs_graphs_dir)
                CombinedVisualizer.save_major_topics_graph(
                    merged_infos, date_key=date_key, output_graphs_dir=date_outputs_graphs_dir)
                print("  ✓ Merged graphs completed")
            except Exception as e:
                print(f"  ✗ Error generating merged graphs: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"✗ Error processing date {date_str}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # テンポラリファイルを削除
            if os.path.exists(tmp_file_name):
                os.remove(tmp_file_name)
