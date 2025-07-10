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
EMB_DIR = config['embeddings_dir']

# KeyBERT モデル初期化（1回だけ）
_kw_model = KeyBERT(model='all-MiniLM-L6-v2')


def _to_numpy(tensor):
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
        # 埋め込みベクトルおよびAttention/Maskをバイナリ保存
        stage_dir = os.path.join(EMB_DIR, f"{col_name_from}_{col_name_to}")
        os.makedirs(stage_dir, exist_ok=True)
        inp = _to_numpy(self.input_tensor)
        out = _to_numpy(self.output_tensor)
        # attention と mask
        att_in = _to_numpy(self.attention_value_lists[0])
        att_out = _to_numpy(self.attention_value_lists[1])
        mask_in = _to_numpy(self.input_padding_flags)
        mask_out = _to_numpy(self.output_padding_flags)
        # サンプルID
        ids = list(range(inp.shape[0]))
        # 保存
        np.save(os.path.join(stage_dir, 'input_embeddings.npy'), inp)
        np.save(os.path.join(stage_dir, 'output_embeddings.npy'), out)
        np.save(os.path.join(stage_dir, 'input_attention.npy'), att_in)
        np.save(os.path.join(stage_dir, 'output_attention.npy'), att_out)
        np.save(os.path.join(stage_dir, 'input_mask.npy'), mask_in)
        np.save(os.path.join(stage_dir, 'output_mask.npy'), mask_out)
        with open(os.path.join(stage_dir, 'sample_ids.json'), 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False)

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
            # UMAP + DBSCAN でクラスタリング
            red = UMAP(n_neighbors=3, init='random',
                       random_state=0).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            cluster_nums.append(labels.tolist())
            cluster_maxs.append(int(labels.max()))
            # 列ごとの散布図出力(従来処理)
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
    def merge_and_recluster(eps, min_samples):
        n_pairs = len(COL_NAMES) - 1
        pair_data = []
        for idx in range(n_pairs):
            cf = COL_NAMES[idx]
            ct = COL_NAMES[idx + 1]
            stage_dir = os.path.join(EMB_DIR, f"{cf}_{ct}")
            # 3D埋め込み・Attention・Maskをロード
            # [n_samples, n_sentences, dim]
            inp_3d = np.load(os.path.join(stage_dir, 'input_embeddings.npy'))
            # [n_samples, n_sentences]
            att_in = np.load(os.path.join(stage_dir, 'input_attention.npy'))
            # [n_samples, n_sentences]
            mask_in = np.load(os.path.join(stage_dir, 'input_mask.npy'))
            # [n_samples, n_sentences, dim]
            out_3d = np.load(os.path.join(stage_dir, 'output_embeddings.npy'))
            # [n_samples, n_sentences]
            att_out = np.load(os.path.join(stage_dir, 'output_attention.npy'))
            # [n_samples, n_sentences]
            mask_out = np.load(os.path.join(stage_dir, 'output_mask.npy'))
            # Attention重み付き和で2D埋め込み化
            inp_emb = np.einsum('ij, ijk -> ik', mask_in * att_in, inp_3d)
            out_emb = np.einsum('ij, ijk -> ik', mask_out * att_out, out_3d)
            # 代表文候補ロード
            with open(os.path.join(TMP_DIR, f"cluster_info_{idx}_{idx+1}.json"), 'r', encoding='utf-8') as f:
                info = json.load(f)
            s_in = info['sentences'][0]
            s_out = info['sentences'][1]
            pair_data.append({
                'in_emb': inp_emb, 'out_emb': out_emb,
                'in_sent': s_in, 'out_sent': s_out
            })
        # ステージリスト構築
        stages = []
        # 第1ステージ（only in）
        stages.append({
            'emb': pair_data[0]['in_emb'],
            'sent': pair_data[0]['in_sent'],
            'col': COL_NAMES[0]
        })
        # 中間ステージ（merge + 代表文候補）
        for i in range(1, n_pairs):
            prev = pair_data[i-1]
            curr = pair_data[i]
            merged_emb = (prev['out_emb'] + curr['in_emb']
                          ) / 2  # [n_samples, dim]
            # [n_samples, 2, dim]
            cand_emb = np.stack([prev['out_emb'], curr['in_emb']], axis=1)
            cand_sent = list(zip(prev['out_sent'], curr['in_sent']))
            stages.append({
                'emb': merged_emb,
                'cand_emb': cand_emb,
                'cand_sent': cand_sent,
                'col': COL_NAMES[i]
            })
        # 最終ステージ（only out）
        stages.append({
            'emb': pair_data[-1]['out_emb'],
            'sent': pair_data[-1]['out_sent'],
            'col': COL_NAMES[-1]
        })

        merged_infos = []
        # 再クラスタリング & 代表文選定 & トピック抽出
        for stage_idx, st in enumerate(stages):
            X = st['emb']  # [n_samples, dim]
            col = st['col']
            # UMAP→DBSCAN
            red = UMAP(n_neighbors=3, init='random',
                       random_state=0).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            # 代表文選定
            if 'cand_emb' in st:
                centers = {}
                for cid in np.unique(labels):
                    idxs = np.where(labels == cid)[0]
                    centers[cid] = X[idxs].mean(axis=0)
                sentences = []
                for j, cid in enumerate(labels):
                    c_emb = centers[cid]
                    cands = st['cand_emb'][j]  # [2, dim]
                    sims = (cands @ c_emb) / (
                        np.linalg.norm(cands, axis=1) * np.linalg.norm(c_emb) + 1e-12)
                    best = sims.argmax()
                    sentences.append(st['cand_sent'][j][best])
            else:
                sentences = st['sent']
            # KeyBERTトピック抽出
            docs_by_cluster = defaultdict(list)
            for cid, s in zip(labels.tolist(), sentences):
                docs_by_cluster[cid].append(s)
            topics = {}
            for cid, docs in docs_by_cluster.items():
                combined = '。'.join(docs)
                kws = _kw_model.extract_keywords(
                    combined, keyphrase_ngram_range=(1, 2),
                    stop_words='english', use_mmr=True, top_n=3
                )
                topics[cid] = [kw[0] for kw in kws]
            merged_infos.append({
                'col': col,
                'clusters': labels.tolist(),
                'topics': topics
            })
        return merged_infos

    @staticmethod
    def draw_merged_bigraph(merged_infos):
        fig, ax = plt.subplots(figsize=(16, 8))
        G = nx.Graph()
        pos, edge_w, labels = {}, Counter(), {}
        # ノード生成
        for stage, info in enumerate(merged_infos):
            col = info['col']
            clusters = info['clusters']
            topics = info['topics']
            for sample_idx, cid in enumerate(clusters):
                node = f"{col}:{cid}:p{stage}"
                G.add_node(node)
                labels[node] = '\n'.join(topics.get(cid, [])) or f"{col}:{cid}"
                pos[node] = (stage, sample_idx)
        # エッジ生成: サンプルIDで繋ぐ
        n_samples = len(merged_infos[0]['clusters'])
        for sample in range(n_samples):
            for stage in range(len(merged_infos) - 1):
                cid1 = merged_infos[stage]['clusters'][sample]
                cid2 = merged_infos[stage+1]['clusters'][sample]
                u = f"{merged_infos[stage]['col']}:{cid1}:p{stage}"
                v = f"{merged_infos[stage+1]['col']}:{cid2}:p{stage+1}"
                edge_w[(u, v)] += 1
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)
        # 描画
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sizes = [300 * G.degree(n, weight='weight') for n in G.nodes()]
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=sizes, width=weights,
                node_color='white', edgecolors='tomato', linewidths=1.5, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', bbox=dict(
            facecolor='white', edgecolor='none', alpha=0.8), ax=ax)
        plt.tight_layout()
        os.makedirs(OUTPUTS_GRAPHS_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUTS_GRAPHS_DIR, 'merged_bigraph.png'))
        plt.show()
