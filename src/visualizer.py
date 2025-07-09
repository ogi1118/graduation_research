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


def _to_numpy(tensor):
    # Tensor かつ CUDA の場合とそうでない場合
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().detach().numpy()
    else:
        return np.array(tensor)


class Visualizer:
    def __init__(self, input_tensor, output_tensor, input_padding_flags,
                 output_padding_flags, sentence_lists, attention_value_lists,
                 device=torch.device('cpu'), col_name_from=None, col_name_to=None):
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
        inp = _to_numpy(self.input_tensor)
        inp_mask = _to_numpy(self.input_padding_flags)
        out = _to_numpy(self.output_tensor)
        out_mask = _to_numpy(self.output_padding_flags)
        att_in = _to_numpy(self.attention_value_lists[0])
        att_out = _to_numpy(self.attention_value_lists[1])

        data_arrays = [inp, out]
        masks = [inp_mask, out_mask]
        atts = [att_in, att_out]

        sentences_list, cluster_nums, cluster_maxs = [], [], []
        for i in range(2):
            emb_list, sent_keys = [], []
            for emb, mask, sents, att in zip(
                    data_arrays[i], masks[i], self.sentence_lists, atts[i]):
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
            # 列ごとの散布図出力
            plt.figure()
            plt.scatter(red[:, 0], red[:, 1], c=labels, alpha=0.8)
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
                (self.col_name_from, self.col_name_to), nums, sents):
            docs_by_cluster = defaultdict(list)
            for cid, sent in zip(clusters, sentences):
                docs_by_cluster[cid].append(sent)
            for cid, docs in docs_by_cluster.items():
                combined = '。'.join(docs)
                kws = _kw_model.extract_keywords(
                    combined, keyphrase_ngram_range=(1, 2),
                    stop_words='english', use_mmr=True, top_n=3
                )
                topics[col][cid] = [kw[0] for kw in kws]
        os.makedirs(TMP_DIR, exist_ok=True)
        idx = COL_NAMES.index(self.col_name_from)
        jdx = COL_NAMES.index(self.col_name_to)
        with open(os.path.join(TMP_DIR, f"cluster_info_{idx}_{jdx}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'col_pair': [self.col_name_from, self.col_name_to],
                'clusters': [list(map(int, nums[0])), list(map(int, nums[1]))],
                'sentences': sents,
                'cluster_max': [int(maxs[0]), int(maxs[1])],
                'topics': topics
            }, f, ensure_ascii=False, indent=2)


class CombinedVisualizer:
    @staticmethod
    def load_cluster_infos():
        infos = []
        for idx in range(len(COL_NAMES) - 1):
            jdx = idx + 1
            fn = os.path.join(TMP_DIR, f"cluster_info_{idx}_{jdx}.json")
            if os.path.isfile(fn):
                with open(fn, encoding='utf-8') as f:
                    infos.append(json.load(f))
        return infos

    @staticmethod
    def draw_combined_bigraph(cluster_infos):
        fig, ax = plt.subplots(figsize=(16, 8))
        G = nx.Graph()
        pos, edge_w, labels = {}, Counter(), {}

        # 各ステージの内部エッジノード定義
        for stage, info in enumerate(cluster_infos):
            cf, ct = info['col_pair']
            cl_f, cl_t = info['clusters']
            topics = info.get('topics', {})
            for f_id, t_id in zip(cl_f, cl_t):
                u = f"{cf}:{f_id}:p{stage}:l"
                v = f"{ct}:{t_id}:p{stage}:r"
                G.add_node(u)
                G.add_node(v)
                # ラベル折返し
                txt_f = '\n'.join(topics.get(cf, {}).get(
                    str(f_id), [])) or f"{cf}:{f_id}"
                txt_t = '\n'.join(topics.get(ct, {}).get(
                    str(t_id), [])) or f"{ct}:{t_id}"
                labels[u] = txt_f
                labels[v] = txt_t
                edge_w[(u, v)] += 1
                # 仮 Y 値
                pos[u] = (stage * 2, 0)
                pos[v] = (stage * 2 + 1, 0)

        # 内部エッジ追加
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)

        # ステージ間クロスエッジ
        cross_w = Counter()
        for stage in range(len(cluster_infos) - 1):
            curr = cluster_infos[stage]
            nxt = cluster_infos[stage + 1]
            col_mid = COL_NAMES[stage + 1]
            n_samples = len(curr['clusters'][0])
            for i in range(n_samples):
                t_id = curr['clusters'][1][i]
                f_next = nxt['clusters'][0][i]
                u = f"{col_mid}:{t_id}:p{stage}:r"
                v = f"{col_mid}:{f_next}:p{stage+1}:l"
                cross_w[(u, v)] += 1
        for (u, v), w in cross_w.items():
            G.add_edge(u, v, weight=w)

        # XごとY均等配置
        x_groups = defaultdict(list)
        for n, (x, _) in pos.items():
            x_groups[x].append(n)
        for x, nodes in x_groups.items():
            N = len(nodes)
            for i, n in enumerate(sorted(nodes)):
                pos[n] = (x, i / (N - 1) if N > 1 else 0.5)

        # 描画
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sizes = [300 * G.degree(n, weight='weight') for n in G.nodes()]
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=sizes,
                width=weights, node_color='tomato', alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # 下部カラムペア表示
        total = len(cluster_infos)
        for i, info in enumerate(cluster_infos):
            cf, ct = info['col_pair']
            x_rel = (2 * i + 0.5) / (2 * total - 1)
            ax.text(x_rel, -0.05, f"{cf}\n→\n{ct}",
                    ha='center', va='top', transform=ax.transAxes)

        os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_GRAPHS_DIR, 'combined_bigraph.png'))
        plt.show()
        print('Saved combined graph to', OUTPUTS_GRAPHS_DIR)
