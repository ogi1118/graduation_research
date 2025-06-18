from dual_attention import DualAttention
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from umap import UMAP
# pip install umap-learn
import seaborn as sns
import torch
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
import csv

class Executer:
    def __init__(self, file_name, col_names) -> None:
        # 初期化処理：ファイル名とcolumn名を設定
        self.encoded_vector_dim = 20 
        self.da = DualAttention(self.encoded_vector_dim)
        self.device = self.da.which_device()
        self.file_name = file_name
        self.col_names = col_names
        self.sentence_lists = None

    def train(self, max_sentence_number=20, epochs=100):
        # DualAttentionを使用してデータを読み込み
        tensors, padding_flags = self.da.read(
            max_sentence_number=max_sentence_number, 
            file_name=self.file_name, 
            col_names=self.col_names
        )
        # トレーニング処理を実行
        encoders, assoc = self.da.train(
            tensors, padding_flags, epochs=epochs
        )
        # SentenceListsを取得
        self.sentence_lists = self.da.get_sentence_lists()
        return encoders, assoc, tensors, padding_flags

    def get_sentence_lists(self):
        # SentenceListsを返す
        return self.sentence_lists

    def which_device(self):
        # 使用しているデバイスを返す
        return self.device


class AttentionEvaluator:
    def __init__(self, encoders, tensors, sentence_lists, device="cpu") -> None:
        # 初期化処理：エンコーダ、テンソル、文リストを設定
        self.encoders = encoders
        for encoder in self.encoders:
            encoder.eval()
        self.tensors = tensors
        self.sentence_lists = sentence_lists
        self.device = device

    def evaluate(self, output_file_name):
        # Attention値を評価し、結果をファイルに書き込む
        with open(output_file_name, mode="w") as f:
            attention_values_list = []
            for i, (encoder, tensor) in enumerate(zip(self.encoders, self.tensors)):
                attention_values = encoder.get_attention(tensor)
                attention_values_list.append(attention_values)
                f.write(f"-----------Column-{i}-----------\n")
                for j in range(len(self.sentence_lists)):
                    sentence_list = self.sentence_lists[j][i]
                    f.write(f"...{j}.................\n")
                    for k in range(len(sentence_list)):
                        f.write(f"{sentence_list[k]}:{self.t2str(attention_values[j,k])}\n")
        print("Done.")
        return attention_values_list

    def t2str(self, t):
        # Attention値を文字列に変換
        if self.device == "cpu":
            result = str(t.detach().numpy())
        else:
            result = str(t.cpu().detach().numpy())
        return result


class Visualizer:
    def __init__(self, tensors, padding_flags, sentence_lists, attention_value_lists, device=torch.device("cpu")) -> None:
        # 初期化処理：テンソル、フラグ、文リスト、Attention値を設定
        self.tensors = tensors
        self.padding_flags = padding_flags
        self.sentence_lists = sentence_lists
        self.attention_value_lists = attention_value_lists
        self.device = device

    def get_clusters(self, eps, min_samples):
        # クラスタリング処理を実行
        array_list = []
        flag_list = []
        attention_value_list = []

        for tensor, flags, attention_values in zip(self.tensors, self.padding_flags, self.attention_value_lists):
            if tensor.is_cuda:
                tensor = tensor.cpu()
                flags = flags.cpu()
                attention_values = attention_values.cpu()
            array_list.append(tensor.detach().numpy())
            flag_list.append(flags.detach().numpy())
            attention_value_list.append(attention_values.detach().numpy())

        sentences_list = []
        cluster_num_list = []
        cluster_id_max_list = []
        for i in range(len(array_list)):
            sentences, data = self.find_description_embeddings(array_list, flag_list, attention_value_list, i)
            reducer = UMAP(n_neighbors=3, init='random', random_state=0)
            projected_data_2d = reducer.fit_transform(data)
            pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(projected_data_2d)
            sentences_list.append(sentences)
            cluster_num_list.append(list(pred))
            cluster_id_max_list.append(np.max(pred))
            for sentence, cluster in zip(sentences, pred):
                print(f"{sentence} :> {i}:{cluster}")
            plt.title(f'Column-{i}')
            plt.scatter(projected_data_2d[:, 0], projected_data_2d[:, 1], c=pred, alpha=0.9)
            plt.colorbar()
            plt.savefig(f'Column-{i}.png')
            plt.clf()
        with open("sentence_file.csv", "w") as f:
            writer = csv.writer(f)
            for row in zip(*sentences_list, *cluster_num_list):
                writer.writerow(row)

        print("done")
        return cluster_num_list, sentences_list, cluster_id_max_list

    def find_description_embeddings(self, emb_array_list, flag_list, attention_value_list, i):
        # 文の埋め込みを計算
        data_list = []
        sentences = []
        for embed_array, flags, sentence_list, attentions in zip(emb_array_list[i], flag_list[i], self.sentence_lists, attention_value_list[i]):
            attention = flags * attentions
            att_emb = attention @ embed_array
            sentence = sentence_list[i] 
            data_list.append(np.expand_dims(att_emb, axis=0))
            sentences.append(sentence[attention.argmax()])
        desc_embedding_data = np.concatenate(data_list, axis=0)
        print(f"# of sentences({i}): {len(sentences)}")
        return sentences, desc_embedding_data

    def get_bigraph(self, eps, min_samples):
        # クラスタリング結果を使用して二部グラフを作成
        cluster_num_list, sentences_list, cluster_id_max_list = self.get_clusters(eps, min_samples)
        bigraph_net = nx.Graph()
        edge_list = []
        alt_edge_list = []
        for edge_tuple in zip(*cluster_num_list):
            edge_list.append(tuple(f"{i}:{val}" for i, val in enumerate(edge_tuple)))
            alt_edge_list.append("/".join(edge_list[-1]))
        edge_counter = Counter(alt_edge_list)
        pos = dict()
        for k in range(len(cluster_id_max_list)):
            bigraph_net.add_nodes_from([f"{k}:{i}" for i in range(-1, cluster_id_max_list[k]+1)], bipartite=k)
            pos.update((f"{k}:{i}", (k+1, i)) for i in range(-1, cluster_id_max_list[k]+1))
        for edge in list(set(edge_list)):
            alt_edge = "/".join(edge)
            bigraph_net.add_edge(*edge, weight=edge_counter[alt_edge])
        edge_weights = [bigraph_net[u][v]["weight"] for u, v in bigraph_net.edges]
        mag = 500
        node_degree = [mag * bigraph_net.degree(weight="weight")[i] for i in bigraph_net.nodes]
        pos_spring = nx.spring_layout(bigraph_net, weight="weight", seed=4)
        for k in range(len(cluster_id_max_list)):
            pos.update((f"{k}:{i}", (k+1, pos_spring[f"{k}:{i}"][1])) for i in range(-1, cluster_id_max_list[k]+1))
        nx.draw(bigraph_net, pos, node_color="red", alpha=0.3, with_labels=True, node_size=node_degree, width=edge_weights)
        plt.savefig("bigraph.png")
        plt.show()


if __name__ == "__main__":
    # ファイル名とカラム名を設定
    file_name = "/home/al22091/graduation_research/dataset20240126.csv"
    col_names = [
        "interesting_facts_or_information",
        "identified_problems_or_needs",
        "data_or_information_to_be_further_explored"
    ]

    # Executerクラスを初期化
    executer = Executer(file_name, col_names)

    # データの読み込みとトレーニングを実行
    encoders, assoc, tensors, padding_flags = executer.train(max_sentence_number=20, epochs=1000)

    # SentenceListsを取得
    sentence_lists = executer.get_sentence_lists()

    # AttentionEvaluatorを初期化して評価を実行
    ae = AttentionEvaluator(encoders, tensors, sentence_lists, device=executer.which_device())
    output_file_name = f"./outputs/[{'_'.join(col_names)}].csv"
    attention_value_lists = ae.evaluate(output_file_name)

    # Visualizerを初期化してクラスタリングと可視化を実行
    vis = Visualizer(tensors, padding_flags, sentence_lists, attention_value_lists, device=executer.which_device())
    #vis.get_clusters(0.5, 3)
    vis.get_bigraph(0.7, 3)


