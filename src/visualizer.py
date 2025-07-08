import os
import json
from collections import Counter, defaultdict
import torch
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.cluster import DBSCAN
from keybert import KeyBERT

# 設定読み込み
from config_loader import config

COL_NAMES = config['col_names']
TMP_DIR = config['tmp_dir']
OUTPUTS_GRAPHS_DIR = config['outputs_graphs_dir']
OUTPUTS_CSVS_DIR = config['outputs_csvs_dir']

# KeyBERT モデル初期化（1回だけ）
_kw_model = KeyBERT(model='all-MiniLM-L6-v2')


class Visualizer:
    def __init__(
        self,
        input_tensor,
        output_tensor,
        input_padding_flags,
        output_padding_flags,
        sentence_lists,
        attention_value_lists,
        device=torch.device('cpu'),
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
            emb_list = []
            sent_keys = []
            for emb, mask, sents, att in zip(data_arrays[i], masks[i], self.sentence_lists, atts[i]):
                weight = mask * att
                vec = weight @ emb
                emb_list.append(vec)
                sent_keys.append(sents[i][weight.argmax()])
            X = np.stack(emb_list)
            sentences_list.append(sent_keys)

            red = UMAP(n_neighbors=3, init='random',
                       random_state=0).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            cluster_nums.append(labels.tolist())
            cluster_maxs.append(int(labels.max()))

            plt.figure()
            plt.scatter(red[:, 0], red[:, 1], c=labels, alpha=0.9)
            plt.colorbar()
            os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
            plt.savefig(os.path.join(
                OUTPUTS_GRAPHS_DIR,
                f"[{self.col_name_from}]2[{self.col_name_to}]-column-{i}.png"
            ))
            plt.close()

        return cluster_nums, sentences_list, cluster_maxs

    def save_cluster_info(self, eps, min_samples):
        nums, sents, maxs = self.get_clusters(eps, min_samples)
        topics = {self.col_name_from: {}, self.col_name_to: {}}
        for col, clusters, sentences in zip(
            (self.col_name_from, self.col_name_to),
            nums,
            sents
        ):
            cluster_docs = defaultdict(list)
            for cid, sent in zip(clusters, sentences):
                cluster_docs[cid].append(sent)
            for cid, docs in cluster_docs.items():
                text = '。'.join(docs)
                kws = _kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,
                    top_n=3
                )
                topics[col][cid] = [kw[0] for kw in kws]

        i = COL_NAMES.index(self.col_name_from)
        j = COL_NAMES.index(self.col_name_to)
        data = {
            'col_pair': [self.col_name_from, self.col_name_to],
            'clusters': [list(map(int, nums[0])), list(map(int, nums[1]))],
            'sentences': sents,
            'cluster_max': [int(maxs[0]), int(maxs[1])],
            'topics': topics
        }
        os.makedirs(TMP_DIR, exist_ok=True)
        path = os.path.join(TMP_DIR, f'cluster_info_{i}_{j}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class CombinedVisualizer:
    @staticmethod
    def load_cluster_infos():
        infos = []
        for idx in range(len(COL_NAMES) - 1):
            jdx = idx + 1
            path = os.path.join(TMP_DIR, f'cluster_info_{idx}_{jdx}.json')
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as f:
                    infos.append(json.load(f))
        return infos

    @staticmethod
    def draw_combined_bigraph(cluster_infos):
        # グラフ全体と軸準備
        fig, ax = plt.subplots(figsize=(16, 8))
        G = nx.Graph()
        pos = {}
        edge_w = Counter()
        labels = {}

        # ノードとエッジ収集
        for idx, info in enumerate(cluster_infos):
            cf, ct = info['col_pair']
            cl_f, cl_t = info['clusters']
            topics = info.get('topics', {})
            for f, t in zip(cl_f, cl_t):
                u = f"{cf}:{f}:p{idx}:l"
                v = f"{ct}:{t}:p{idx}:r"
                G.add_node(u)
                G.add_node(v)
                topic_f = '\n'.join(topics.get(cf, {}).get(str(f), []))
                topic_t = '\n'.join(topics.get(ct, {}).get(str(t), []))
                labels[u] = topic_f if topic_f else f"{cf}:{f}"
                labels[v] = topic_t if topic_t else f"{ct}:{t}"
                edge_w[(u, v)] += 1
                pos[u] = (idx*2, f)
                pos[v] = (idx*2+1, t)

        # エッジ追加
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)

        # y正規化
        ys = [y for _, y in pos.values()]
        y0, y1 = min(ys), max(ys)
        for n, (x, y) in pos.items():
            pos[n] = (x, (y - y0)/(y1 - y0) if y1 > y0 else 0.5)

        # ノード・エッジ描画
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sizes = [300 * G.degree(n, weight='weight') for n in G.nodes()]
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=sizes,
                width=weights, node_color='tomato', alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # カラムペア表示（下部）
        total_pairs = len(cluster_infos)
        for idx, info in enumerate(cluster_infos):
            cf, ct = info['col_pair']
            x_rel = (2*idx + 0.5) / (2*total_pairs - 1)
            ax.text(x_rel, -0.05, f"{cf} → {ct}",
                    ha='center', va='top', transform=ax.transAxes)

        # レイアウト・保存
        os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_GRAPHS_DIR, 'combined_bigraph.png'))
        plt.show()
        print('Saved combined graph to ' + OUTPUTS_GRAPHS_DIR)
