import os
import json
from collections import Counter, defaultdict
import torch
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
                 device=torch.device('cpu'), col_name_from=None, col_name_to=None,
                 date_key=None, output_graphs_dir=None, embeddings_dir=None, tmp_dir=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.input_padding_flags = input_padding_flags
        self.output_padding_flags = output_padding_flags
        self.sentence_lists = sentence_lists
        self.attention_value_lists = attention_value_lists
        self.device = device
        self.col_name_from = col_name_from
        self.col_name_to = col_name_to
        self.date_key = date_key
        self.output_graphs_dir = output_graphs_dir or OUTPUTS_GRAPHS_DIR
        self.embeddings_dir = embeddings_dir or EMB_DIR
        self.tmp_dir = tmp_dir or TMP_DIR

        # 埋め込みベクトルおよびAttention/Maskをバイナリ保存
        stage_dir = os.path.join(
            self.embeddings_dir, f"{col_name_from}_{col_name_to}")
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
                       random_state=42).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            cluster_nums.append(labels.tolist())
            cluster_maxs.append(int(labels.max()))
            # 列ごとの散布図出力(従来処理)
            plt.figure()
            plt.scatter(red[:, 0], red[:, 1], c=labels, alpha=0.8)
            plt.colorbar()
            os.makedirs(self.output_graphs_dir, exist_ok=True)
            plt.savefig(os.path.join(
                self.output_graphs_dir,
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
        os.makedirs(self.tmp_dir, exist_ok=True)
        idx = COL_NAMES.index(self.col_name_from)
        jdx = COL_NAMES.index(self.col_name_to)
        with open(os.path.join(self.tmp_dir, f"cluster_info_{idx}_{jdx}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'col_pair': [self.col_name_from, self.col_name_to],
                'clusters': [list(map(int, nums[0])), list(map(int, nums[1]))],
                'sentences': sents,
                'cluster_max': [int(maxs[0]), int(maxs[1])],
                'topics': topics
            }, f, ensure_ascii=False, indent=2)


class CombinedVisualizer:
    @staticmethod
    def load_cluster_infos(tmp_dir=None):
        tmp_dir = tmp_dir or TMP_DIR
        infos = []
        for idx in range(len(COL_NAMES) - 1):
            jdx = idx + 1
            fn = os.path.join(tmp_dir, f"cluster_info_{idx}_{jdx}.json")
            if os.path.isfile(fn):
                with open(fn, encoding='utf-8') as f:
                    infos.append(json.load(f))
        return infos

    @staticmethod
    def merge_and_recluster(eps, min_samples, date_key=None, tmp_dir=None, embeddings_dir=None):
        tmp_dir = tmp_dir or TMP_DIR
        embeddings_dir = embeddings_dir or EMB_DIR

        n_pairs = len(COL_NAMES) - 1
        pair_data = []
        for idx in range(n_pairs):
            cf = COL_NAMES[idx]
            ct = COL_NAMES[idx + 1]
            stage_dir = os.path.join(embeddings_dir, f"{cf}_{ct}")
            inp_3d = np.load(os.path.join(stage_dir, 'input_embeddings.npy'))
            att_in = np.load(os.path.join(stage_dir, 'input_attention.npy'))
            mask_in = np.load(os.path.join(stage_dir, 'input_mask.npy'))
            out_3d = np.load(os.path.join(stage_dir, 'output_embeddings.npy'))
            att_out = np.load(os.path.join(stage_dir, 'output_attention.npy'))
            mask_out = np.load(os.path.join(stage_dir, 'output_mask.npy'))
            inp_emb = np.einsum('ij, ijk -> ik', mask_in * att_in, inp_3d)
            out_emb = np.einsum('ij, ijk -> ik', mask_out * att_out, out_3d)
            with open(os.path.join(tmp_dir, f"cluster_info_{idx}_{idx+1}.json"), 'r', encoding='utf-8') as f:
                info = json.load(f)
            s_in = info['sentences'][0]
            s_out = info['sentences'][1]
            pair_data.append({
                'in_emb': inp_emb, 'out_emb': out_emb,
                'in_sent': s_in, 'out_sent': s_out
            })
        stages = []
        stages.append({
            'emb': pair_data[0]['in_emb'],
            'sent': pair_data[0]['in_sent'],
            'col': COL_NAMES[0]
        })
        for i in range(1, n_pairs):
            prev = pair_data[i-1]
            curr = pair_data[i]
            merged_emb = (prev['out_emb'] + curr['in_emb']) / 2
            cand_emb = np.stack([prev['out_emb'], curr['in_emb']], axis=1)
            cand_sent = list(zip(prev['out_sent'], curr['in_sent']))
            stages.append({
                'emb': merged_emb,
                'cand_emb': cand_emb,
                'cand_sent': cand_sent,
                'col': COL_NAMES[i]
            })
        stages.append({
            'emb': pair_data[-1]['out_emb'],
            'sent': pair_data[-1]['out_sent'],
            'col': COL_NAMES[-1]
        })
        merged_infos = []
        for stage_idx, st in enumerate(stages):
            X = st['emb']
            col = st['col']
            red = UMAP(n_neighbors=3, init='random',
                       random_state=42).fit_transform(X)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(red)
            if 'cand_emb' in st:
                centers = {}
                for cid in np.unique(labels):
                    idxs = np.where(labels == cid)[0]
                    centers[cid] = X[idxs].mean(axis=0)
                sentences = []
                for j, cid in enumerate(labels):
                    c_emb = centers[cid]
                    cands = st['cand_emb'][j]
                    sims = (cands @ c_emb) / (np.linalg.norm(cands,
                                                             axis=1) * np.linalg.norm(c_emb) + 1e-12)
                    best = sims.argmax()
                    sentences.append(st['cand_sent'][j][best])
            else:
                sentences = st['sent']
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
                'emb': X,
                'clusters': labels.tolist(),
                'topics': topics
            })
        return merged_infos

    @staticmethod
    def save_major_topics_graph(merged_infos, node_size_factor=300, edge_width_factor=1.0, date_key=None, output_graphs_dir=None):
        """
        merged_graphと全く同じグラフを描画。ただし各ステージで最大ノード（サンプル数）以外は
        白色（縁なし）で表示して、視覚的に最大ノードのみをハイライト
        """
        output_graphs_dir = output_graphs_dir or OUTPUTS_GRAPHS_DIR

        fig, ax = plt.subplots(figsize=(16, 12))
        G = nx.Graph()
        pos, edge_w, labels = {}, Counter(), {}

        # 全ステージの全クラスタIDを集計してグローバルなY座標マッピングを作成
        all_unique_cids = set()
        for info in merged_infos:
            clusters = info['clusters']
            all_unique_cids.update(c for c in clusters if c != -1)

        # 全クラスタに対してY座標を割り当て
        sorted_all_cids = sorted(all_unique_cids)
        y_max = 10
        y_positions_global = np.linspace(0, y_max, len(
            sorted_all_cids)) if sorted_all_cids else []
        global_cid_to_y = {cid: y_positions_global[i]
                           for i, cid in enumerate(sorted_all_cids)}

        # ノード生成（ノイズ-1を除外）
        node_degrees = {}  # ノードのサイズ情報を保存
        major_nodes_per_stage = {}  # 各ステージの最大ノード

        for stage, info in enumerate(merged_infos):
            col = info['col']
            clusters = info['clusters']
            topics = info['topics']

            # ステージ内での最大サンプル数を追跡
            stage_samples = []

            for sample_idx, cid in enumerate(clusters):
                if cid == -1:  # ノイズを除外
                    continue
                node = f"{col}:{cid}:p{stage}"
                G.add_node(node)
                # デフォルトは白色で初期化
                G.nodes[node]['color'] = 'white'
                labels[node] = '\n'.join(topics.get(cid, [])) or f"{col}:{cid}"
                # クラスタに属するサンプル数でノードサイズを計算
                cid_count = sum(1 for c in clusters if c == cid)
                node_degrees[node] = cid_count
                stage_samples.append((node, cid_count))
                # グローバルなY座標マッピングを使用（全ステージで統一）
                pos[node] = (stage, global_cid_to_y[cid])

            # 各ステージの最大ノードを特定して色をsalmonに変更
            if stage_samples:
                max_node, max_count = max(stage_samples, key=lambda x: x[1])
                major_nodes_per_stage[stage] = max_node
                G.nodes[max_node]['color'] = 'salmon'
                print(
                    f"[DEBUG major] Stage {stage}: max_node={max_node}, sample_count={max_count}")

        # エッジ生成: サンプルIDでノードを繋ぐ（ノイズ-1を除外）
        n_samples = len(merged_infos[0]['clusters'])
        for sample in range(n_samples):
            for stage in range(len(merged_infos) - 1):
                cid1 = merged_infos[stage]['clusters'][sample]
                cid2 = merged_infos[stage + 1]['clusters'][sample]
                if cid1 == -1 or cid2 == -1:  # ノイズが含まれている場合は除外
                    continue
                u = f"{merged_infos[stage]['col']}:{cid1}:p{stage}"
                v = f"{merged_infos[stage + 1]['col']}:{cid2}:p{stage + 1}"
                edge_w[(u, v)] += 1

        # 辺追加
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)

        # ノードサイズとエッジ幅をデータ量に応じて設定
        node_sizes = [node_size_factor *
                      G.degree(n, weight='weight') for n in G.nodes()]
        edge_widths = [edge_width_factor * G[u][v]['weight']
                       for u, v in G.edges()]

        # 最大ノード間のエッジのみ灰色、その他は白色に設定
        edge_color_list = []
        for u, v in G.edges():
            # 両ノードがサーモン色（最大ノード）の場合のみ灰色
            if G.nodes[u]['color'] == 'salmon' and G.nodes[v]['color'] == 'salmon':
                edge_color_list.append('gray')
            else:
                edge_color_list.append('white')

        # グラフ描画
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=False,
            node_size=node_sizes,
            node_color=[G.nodes[n]['color'] for n in G.nodes()],
            width=edge_widths,
            edgecolors=['tomato' if G.nodes[n]['color'] ==
                        'salmon' else 'none' for n in G.nodes()],
            edge_color=edge_color_list,
            linewidths=[0.5 if G.nodes[n]['color'] ==
                        'salmon' else 0 for n in G.nodes()],
            alpha=0.9
        )

        # ラベルはsalmonノード（最大ノード）のみ表示
        labels_to_display = {n: labels[n] if G.nodes[n]['color'] == 'salmon' else ''
                             for n in G.nodes()}

        nx.draw_networkx_labels(
            G,
            pos,
            labels_to_display,
            font_size=20,
            font_color='black',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      alpha=0.2, boxstyle='round,pad=0.3'),
            ax=ax
        )
        plt.tight_layout()
        os.makedirs(output_graphs_dir, exist_ok=True)

        # ファイル名に日付を含める
        filename = f"major_topics{f'_{date_key}' if date_key else ''}.png"
        plt.savefig(os.path.join(output_graphs_dir, filename))
        plt.close()

    @staticmethod
    def draw_merged_graph(merged_infos, node_size_factor=300, edge_width_factor=1.0, date_key=None, output_graphs_dir=None):
        """
        merged_infos: List[Dict] with keys:
        - 'col': str
        - 'clusters': List[int]
        - 'topics': Dict[int, List[str]]
        - 'outlier_flags': List[bool]
        node_size_factor: ノードサイズ倍率
        edge_width_factor: エッジ太さ倍率

        デバッグ情報：各ステージでのノードサイズの最大値を出力
        """
        output_graphs_dir = output_graphs_dir or OUTPUTS_GRAPHS_DIR

        fig, ax = plt.subplots(figsize=(16, 12))
        G = nx.Graph()
        pos, edge_w, labels = {}, Counter(), {}

        # 全ステージの全クラスタIDを集計してグローバルなY座標マッピングを作成
        all_unique_cids = set()
        for info in merged_infos:
            clusters = info['clusters']
            all_unique_cids.update(c for c in clusters if c != -1)

        # 全クラスタに対してY座標を割り当て
        sorted_all_cids = sorted(all_unique_cids)
        y_max = 10
        y_positions_global = np.linspace(0, y_max, len(
            sorted_all_cids)) if sorted_all_cids else []
        global_cid_to_y = {cid: y_positions_global[i]
                           for i, cid in enumerate(sorted_all_cids)}

        # ノード生成（ノイズ-1を除外）
        node_degrees = {}  # ノードのサイズ情報を保存（後でラベル表示判定に使用）
        stage_max_nodes = {}  # 各ステージの最大サイズノードを記録（デバッグ用）

        for stage, info in enumerate(merged_infos):
            col = info['col']
            clusters = info['clusters']
            topics = info['topics']
            flags = info.get('outlier_flags', [False] * len(clusters))

            # ステージ内での最大サンプル数を追跡
            stage_samples = []

            for sample_idx, cid in enumerate(clusters):
                if cid == -1:  # ノイズを除外
                    continue
                node = f"{col}:{cid}:p{stage}"
                G.add_node(node)
                G.nodes[node]['color'] = 'salmon'
                labels[node] = '\n'.join(topics.get(cid, [])) or f"{col}:{cid}"
                # クラスタに属するサンプル数でノードサイズを計算
                cid_count = sum(1 for c in clusters if c == cid)
                node_degrees[node] = cid_count
                stage_samples.append((node, cid_count))
                # グローバルなY座標マッピングを使用（全ステージで統一）
                pos[node] = (stage, global_cid_to_y[cid])

            # デバッグ：各ステージの最大ノードを記録
            if stage_samples:
                max_node, max_count = max(stage_samples, key=lambda x: x[1])
                stage_max_nodes[stage] = (max_node, max_count)
                print(
                    f"[DEBUG merged_graph] Stage {stage} ({col}): max_node={max_node}, sample_count={max_count}")

        # エッジ生成: サンプルIDでノードを繋ぐ（ノイズ-1を除外）
        n_samples = len(merged_infos[0]['clusters'])
        for sample in range(n_samples):
            for stage in range(len(merged_infos) - 1):
                cid1 = merged_infos[stage]['clusters'][sample]
                cid2 = merged_infos[stage + 1]['clusters'][sample]
                if cid1 == -1 or cid2 == -1:  # ノイズが含まれている場合は除外
                    continue
                u = f"{merged_infos[stage]['col']}:{cid1}:p{stage}"
                v = f"{merged_infos[stage + 1]['col']}:{cid2}:p{stage + 1}"
                edge_w[(u, v)] += 1

        # 辺追加
        for (u, v), w in edge_w.items():
            G.add_edge(u, v, weight=w)

        # ノードサイズとエッジ幅をデータ量に応じて設定
        node_sizes = [node_size_factor *
                      G.degree(n, weight='weight') for n in G.nodes()]
        edge_widths = [edge_width_factor * G[u][v]['weight']
                       for u, v in G.edges()]

        # グラフ描画
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=False,
            node_size=node_sizes,
            node_color=[G.nodes[n]['color'] for n in G.nodes()],
            width=edge_widths,
            edgecolors='tomato',
            edge_color='gray',
            linewidths=0.5,
            alpha=0.9
        )

        # ラベル重複判定と描画処理
        # ノードのバウンディングボックスから重複を検出し、小さいノードのラベルを非表示にする
        # labels_to_display = labels.copy()
        # node_pos_list = [(n, pos[n]) for n in G.nodes()]
        #
        # # y座標が近いノード同士を検出
        # threshold = y_max * 0.1  # 重複判定の閾値
        # for i, (node_i, pos_i) in enumerate(node_pos_list):
        #     for node_j, pos_j in node_pos_list[i+1:]:
        #         # 同じステージのノード同士は比較しない（エッジでつながっているため）
        #         x_i, y_i = pos_i
        #         x_j, y_j = pos_j
        #
        #         # y座標が近く、x座標が異なる場合をチェック
        #         if x_i != x_j and abs(y_i - y_j) < threshold:
        #             # ノードサイズが小さい方のラベルを非表示
        #             size_i = node_degrees.get(node_i, 1)
        #             size_j = node_degrees.get(node_j, 1)
        #             if size_i < size_j:
        #                 labels_to_display[node_i] = ''
        #             elif size_j < size_i:
        #                 labels_to_display[node_j] = ''

        # すべてのラベルを表示（重複削除なし）
        labels_to_display = labels

        nx.draw_networkx_labels(
            G,
            pos,
            labels_to_display,
            font_size=20,
            font_color='black',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      alpha=0.2, boxstyle='round,pad=0.3'),
            ax=ax
        )
        plt.tight_layout()
        os.makedirs(output_graphs_dir, exist_ok=True)

        # ファイル名に日付を含める
        filename = f"merged_graph{f'_{date_key}' if date_key else ''}.png"
        plt.savefig(os.path.join(output_graphs_dir, filename))
        plt.close()
