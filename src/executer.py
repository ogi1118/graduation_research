import csv
import os
import json
from collections import Counter
from networkx.algorithms import bipartite
import networkx as nx
import torch
import seaborn as sns
from umap import UMAP
import matplotlib.pyplot as plt
from dual_attention import DualAttention
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Qt5Agg')
# pip install umap-learn


class Executer:
    def __init__(self, file_name, col_name_from, col_name_to) -> None:
        self.encoded_vector_dim = 20
        self.da = DualAttention(self.encoded_vector_dim)
        self.device = self.da.which_device()
        self.file_name, self.col_name_from, self.col_name_to = file_name, col_name_from, col_name_to
        self.sentence_lists = None

    def train(self, max_sentence_number=20, epochs=100):
        input_tensor, input_padding_flags, output_tensor, output_padding_flags = self.da.read(max_sentence_number=max_sentence_number, file_name=self.file_name,
                                                                                              col_name_from=self.col_name_from, col_name_to=self.col_name_to)
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


class Visualizer:
    def __init__(self, input_tensor, output_tensor, input_padding_flags, output_padding_flags, sentence_lists, attention_value_lists, device=torch.device("cpu"), col_name_from=None, col_name_to=None) -> None:
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
        array_list = [[], []]
        flag_list = [[], []]
        attention_value_list = [[], []]

        if self.input_tensor.is_cuda:
            self.input_tensor = self.input_tensor.cpu()
            self.input_padding_flags = self.input_padding_flags.cpu()
            self.output_tensor = self.output_tensor.cpu()
            self.output_padding_flags = self.output_padding_flags.cpu()
            self.attention_value_lists[0] = self.attention_value_lists[0].cpu()
            self.attention_value_lists[1] = self.attention_value_lists[1].cpu()
        array_list[0] = self.input_tensor.detach().numpy()
        flag_list[0] = self.input_padding_flags.detach().numpy()
        attention_value_list[0] = self.attention_value_lists[0].detach(
        ).numpy()
        array_list[1] = self.output_tensor.detach().numpy()
        flag_list[1] = self.output_padding_flags.detach().numpy()
        attention_value_list[1] = self.attention_value_lists[1].detach(
        ).numpy()

        sentences_list = []
        cluster_num_list = []
        cluster_id_max_list = []
        for i in range(2):
            sentences, data = self.find_description_embeddings(
                array_list, flag_list, attention_value_list, i)
            reducer = UMAP(n_neighbors=3, init='random', random_state=0)  # 5
            projected_data_2d = reducer.fit_transform(data)
            pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                projected_data_2d)
            sentences_list.append(sentences)
            cluster_num_list.append(list(pred))
            cluster_id_max_list.append(np.max(pred))
            # for sentence, cluster in zip(sentences, pred):
            # print(sentence+" :> "+str(i)+":"+str(cluster))
            plt.title('Column-{}'.format(str(i)))
            plt.scatter(
                projected_data_2d[:, 0], projected_data_2d[:, 1], c=pred, alpha=0.9)
            plt.colorbar()
            plt.savefig('../outputs/graphs/[{}]2[{}]-column-{}.png'.format(
                self.col_name_from, self.col_name_to, str(i)))
            plt.clf()
        with open("../outputs/csvs/[{}]2[{}]-sentence_file.csv".format(self.col_name_from, self.col_name_to), "w") as f:
            writer = csv.writer(f)
            for sentence_from, cluster_from, sentence_to, cluster_to in zip(sentences_list[0], cluster_num_list[0], sentences_list[1], cluster_num_list[1]):
                writer.writerow([cluster_from, sentence_from,
                                cluster_to,  sentence_to])

        print("done")
        return cluster_num_list, sentences_list, cluster_id_max_list

    def find_description_embeddings(self, emb_array_list, flag_list, attention_value_list, i):
        data_list = []
        sentences = []
        for embed_array, flags, sentence_list, attentions in zip(emb_array_list[i], flag_list[i], self.sentence_lists, attention_value_list[i]):
            attention = flags * attentions
            att_emb = attention @ embed_array
            sentence = sentence_list[i]
            data_list.append(np.expand_dims(att_emb, axis=0))
            sentences.append(sentence[attention.argmax()])
        desc_embedding_data = np.concatenate(data_list, axis=0)
        # print("# of sentences({}): {}".format(i, len(sentences)))
        return sentences, desc_embedding_data

    def get_bigraph(self, eps, min_samples):
        # 1) クラスタ情報取得
        cluster_num_list, _, cluster_id_max_list = self.get_clusters(
            eps, min_samples)

        # 2) グラフ作成
        G = nx.Graph()
        col_names = [self.col_name_from, self.col_name_to]

        # 3) ノード追加（列名:クラスタID の形式）
        for idx, col in enumerate(col_names):
            for cid in range(-1, cluster_id_max_list[idx] + 1):
                G.add_node(f"{col}:{cid}", bipartite=idx)

        # 4) エッジ集計
        edge_counter = Counter()
        for f, t in zip(cluster_num_list[0], cluster_num_list[1]):
            node_f = f"{self.col_name_from}:{f}"
            node_t = f"{self.col_name_to}:{t}"
            edge_counter[(node_f, node_t)] += 1

        # 5) エッジ追加
        for (node_f, node_t), weight in edge_counter.items():
            G.add_edge(node_f, node_t, weight=weight)

        # 6) 初期位置設定 (x 座標は 0 or 1, y はクラスタID)
        pos = {}
        for idx, col in enumerate(col_names):
            for cid in range(-1, cluster_id_max_list[idx] + 1):
                pos[f"{col}:{cid}"] = (idx, cid)

        # 7) Spring レイアウトで y 座標のみ微調整
        spring_pos = nx.spring_layout(G, weight="weight", seed=4)
        for node, (x0, _) in pos.items():
            pos[node] = (x0, spring_pos[node][1])

        # 8) 描画パラメータ準備
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        degrees = [G.degree(n, weight="weight") for n in G.nodes()]
        sizes = [500 * deg for deg in degrees]

        # 9) 描画
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=sizes,
            width=weights,
            node_color="red",
            alpha=0.3,
        )
        plt.savefig(
            f"../outputs/graphs/[{self.col_name_from}]2[{self.col_name_to}]-bigraph.png"
        )
        plt.show()

    def save_cluster_info(self, eps, min_samples):
        cluster_num_list, sentences_list, cluster_id_max_list = self.get_clusters(
            eps, min_samples)

        print("col_name_from: {}, col_name_to: {}".format(
            self.col_name_from, self.col_name_to))
        print("cluster_num_list: {}".format(cluster_num_list))
        print("sentences_list: {}".format(sentences_list))
        print("cluster_id_max_list: {}".format(cluster_id_max_list))

        data = {
            "col_pair": [col_name_from, col_name_to],
            "clusters": [list(map(int, cluster_num_list[0])),
                         list(map(int, cluster_num_list[1]))],
            "sentences": sentences_list,
            "cluster_max": [int(cluster_id_max_list[0]), int(cluster_id_max_list[1])],
        }
        with open("../tmp/cluster_info_{}_{}.json".format(col_name_from, col_name_to), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    file_name = "/home/al22091/graduation_research/dataset20240126.csv"
    col_names = ["curiosity", "interesting_facts_or_information",
                 "identified_problems_or_needs"]
    os.makedirs("../outputs", exist_ok=True)
    os.makedirs("../tmp", exist_ok=True)
    os.makedirs("../outputs/graphs", exist_ok=True)
    os.makedirs("../outputs/csvs", exist_ok=True)
    for i in range(len(col_names) - 1):
        j = i + 1
        col_name_from = col_names[i]
        col_name_to = col_names[j]
        print("col_name_from: {}, col_name_to: {}".format(
            col_name_from, col_name_to))
        executer = Executer(file_name, col_name_from, col_name_to)
        in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(
            max_sentence_number=20, epochs=1000)
        sentence_lists = executer.get_sentence_lists()
        ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor,
                                output_tensor, sentence_lists, device=executer.which_device())
        output_file_name = "../outputs/csvs/[" + \
            col_name_from + "]2[" + col_name_to + "].csv"
        attention_value_lists = ae.evaluate(output_file_name)

        vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags,
                         sentence_lists, attention_value_lists, device=executer.which_device(), col_name_from=col_name_from, col_name_to=col_name_to)
        vis.save_cluster_info(eps=0.7, min_samples=3)
        vis.get_bigraph(0.7, 3)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # executer = Executer(file_name, col_name_from, col_name_to)
    # in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(max_sentence_number=20, epochs=1000)
    # sentence_lists = executer.get_sentence_lists()
    # ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists, device=executer.which_device())
    # output_file_name = "./outputs/[" + col_name_from + "]2[" + col_name_to + "].csv"
    # attention_value_lists = ae.evaluate(output_file_name)
    # vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags, sentence_lists, attention_value_lists, device=executer.which_device())
    # #vis.get_clusters(0.5, 3)
    # vis.get_bigraph(0.7, 3) #0.5, 3
