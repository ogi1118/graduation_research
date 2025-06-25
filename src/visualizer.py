import os
import json
from collections import Counter
import torch
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.cluster import DBSCAN

# 設定読み込み
from config_loader import config

COL_NAMES = config["col_names"]
TMP_DIR = config["tmp_dir"]
OUTPUTS_GRAPHS_DIR = config["outputs_graphs_dir"]
OUTPUTS_CSVS_DIR = config["outputs_csvs_dir"]


class Visualizer:
    def __init__(
        self,
        input_tensor,
        output_tensor,
        input_padding_flags,
        output_padding_flags,
        sentence_lists,
        attention_value_lists,
        device=torch.device("cpu"),
        col_name_from=None,
        col_name_to=None,
    ):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.input_padding_flags = input_padding_flags
        self.output_padding_flags = output_padding_flags
        self.sentence_lists = sentence_lists
        self.attention_value_lists = attention_value_lists
        self.device = device
        self.col_name_from = col_name_from
        self.col_name_to = col_name_to

    def get_clusters(self, eps, min_samples):
        # テンソルと注意重みを numpy に変換
        def to_numpy(tensor):
            return tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()

        inp = to_numpy(self.input_tensor)
        inp_mask = to_numpy(self.input_padding_flags)
        out = to_numpy(self.output_tensor)
        out_mask = to_numpy(self.output_padding_flags)
        att_in = to_numpy(self.attention_value_lists[0])
        att_out = to_numpy(self.attention_value_lists[1])

        data_arrays = [inp, out]
        masks = [inp_mask, out_mask]
        atts = [att_in, att_out]

        sentences_list = []
        cluster_nums = []
        cluster_maxs = []

        for i in range(2):
            # Attention 結合→埋め込み取得
            emb_list = []
            sent_keys = []
            for emb, mask, sents, att in zip(data_arrays[i], masks[i], self.sentence_lists, atts[i]):
                weight = mask * att
                vec = weight @ emb
                emb_list.append(vec)
                sent_keys.append(sents[i][weight.argmax()])
            X = np.stack(emb_list)
            sentences_list.append(sent_keys)

            # 次元削減 + DBSCAN
            red = UMAP(n_neighbors=3, init='random',
                       random_state=0).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            cluster_nums.append(labels.tolist())
            cluster_maxs.append(int(labels.max()))

            # クラスタ可視化グラフ出力
            plt.figure()
            plt.scatter(red[:, 0], red[:, 1], c=labels, alpha=0.9)
            plt.colorbar()
            os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
            plt.savefig(os.path.join(
                OUTPUTS_GRAPHS_DIR,
                f"[{self.col_name_from}]2[{self.col_name_to}]-column-{i}.png"
            ))
            plt.close()

        # 対応表を CSV 保存
        os.makedirs(OUTPUTS_CSVS_DIR, exist_ok=True)
        csv_path = os.path.join(
            OUTPUTS_CSVS_DIR,
            f"[{self.col_name_from}]2[{self.col_name_to}]-sentences.csv"
        )
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for a, s_a, b, s_b in zip(
                cluster_nums[0], sentences_list[0], cluster_nums[1], sentences_list[1]
            ):
                writer.writerow([a, s_a, b, s_b])

        return cluster_nums, sentences_list, cluster_maxs

    def get_bigraph(self, eps, min_samples):
        # クラスタ情報取得
        nums, _, maxs = self.get_clusters(eps, min_samples)
        G = nx.Graph()
        # ノード追加
        for idx, col in enumerate((self.col_name_from, self.col_name_to)):
            for cid in range(-1, maxs[idx]+1):
                G.add_node(f"{col}:{cid}", bipartite=idx)
        # エッジ追加
        cnt = Counter((f"{self.col_name_from}:{f}", f"{self.col_name_to}:{t}")
                      for f, t in zip(nums[0], nums[1]))
        for (u, v), w in cnt.items():
            G.add_edge(u, v, weight=w)
        # レイアウト
        pos = {}
        xmap = {col: i for i, col in enumerate(
            (self.col_name_from, self.col_name_to))}
        for n in G.nodes():
            col, cid = n.split(':')
            pos[n] = (xmap[col], int(cid))
        # y正規化
        ys = [y for _, y in pos.values()]
        y0, y1 = min(ys), max(ys)
        for n, (x, y) in pos.items():
            pos[n] = (x, (y-y0)/(y1-y0) if y1 > y0 else 0.5)
        # 描画
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sizes = [300*G.degree(n, weight='weight') for n in G.nodes()]
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=sizes,
                width=weights, node_color='tomato', alpha=0.6, font_size=8)
        os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUTS_GRAPHS_DIR,
            f"[{self.col_name_from}]2[{self.col_name_to}]-bigraph.png"
        ))
        plt.close()

    def save_cluster_info(self, eps, min_samples):
        # クラスタ情報取得
        nums, sents, maxs = self.get_clusters(eps, min_samples)
        # i,j インデックス
        i = COL_NAMES.index(self.col_name_from)
        j = COL_NAMES.index(self.col_name_to)
        data = {
            "col_pair": [self.col_name_from, self.col_name_to],
            "clusters": [list(map(int, nums[0])), list(map(int, nums[1]))],
            "sentences": sents,
            "cluster_max": [int(maxs[0]), int(maxs[1])]
        }
        os.makedirs(TMP_DIR, exist_ok=True)
        path = os.path.join(TMP_DIR, f"cluster_info_{i}_{j}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class CombinedVisualizer:
    @staticmethod
    def load_cluster_infos():
        infos = []
        for fn in os.listdir(TMP_DIR):
            if not fn.startswith("cluster_info_") or not fn.endswith('.json'):
                continue
            parts = fn.rstrip('.json').split('_')[-2:]
            i, j = map(int, parts)
            if abs(i-j) != 1:
                continue
            with open(os.path.join(TMP_DIR, fn), encoding='utf-8') as f:
                infos.append(json.load(f))
        return infos

    @staticmethod
    def draw_combined_bigraph(cluster_infos):
        G = nx.Graph()
        pos = {}
        labels = {}
        edge_w = Counter()
        # ノードとエッジ収集
        for idx, info in enumerate(cluster_infos):
            cf, ct = info['col_pair']
            cl_f, cl_t = info['clusters']
            for f, t in zip(cl_f, cl_t):
                u = f"{cf}:{f}:p{idx}:l"
                v = f"{ct}:{t}:p{idx}:r"
                G.add_node(u)
                G.add_node(v)
                labels[u] = f"{cf}:{f}"
                labels[v] = f"{ct}:{t}"
                edge_w[(u, v)] += 1
                pos[u] = (idx*2,   f)
                pos[v] = (idx*2+1, t)
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)
        # y正規化
        ys = [y for _, y in pos.values()]
        y0, y1 = min(ys), max(ys)
        for n, (x, y) in pos.items():
            pos[n] = (x, (y-y0)/(y1-y0) if y1 > y0 else 0.5)
        # 描画
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sizes = [300*G.degree(n, weight='weight') for n in G.nodes()]
        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=False, node_size=sizes,
                width=weights, node_color='tomato', alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_GRAPHS_DIR, "combined_bigraph.png"))
        plt.show()
        print("Saved combined graph to " + OUTPUTS_GRAPHS_DIR)
