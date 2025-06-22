from dual_attention import DualAttention
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
import torch
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
import csv
import os

class Executer:
    def __init__(self, file_name) -> None:
        self.encoded_vector_dim = 20 
        self.da = DualAttention(self.encoded_vector_dim)
        self.device = self.da.which_device()
        self.file_name = file_name
        self.sentence_lists = None

    def train(self, col_name_from, col_name_to, max_sentence_number=20, epochs=100):
        input_tensor, input_padding_flags, output_tensor, output_padding_flags = self.da.read(
            max_sentence_number=max_sentence_number,
            file_name=self.file_name,
            col_name_from=col_name_from,
            col_name_to=col_name_to
        )
        in_encoder, out_encoder, assoc = self.da.train(
            input_tensor, input_padding_flags, output_tensor, output_padding_flags, epochs=epochs
        )
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
            in_or_out = ["-----------in-----------", "-----------out-----------"]
            attention_values_list = []
            for i, (encoder, sentence_tensor) in enumerate(zip(self.encoders, self.sentence_tensors)):
                attention_values = encoder.get_attention(sentence_tensor)
                attention_values_list.append(attention_values)
                f.write(in_or_out[i] + "\n")
                for j in range(len(self.sentence_lists)):
                    print(f"Processing {in_or_out[i]} {j}th sentence list...")
                    sentence_list = self.sentence_lists[i][j]
                    f.write("..." + str(j) + "................." + "\n")
                    for k in range(len(sentence_list)):
                        f.write(sentence_list[k] + ":" + self.t2str(attention_values[j, k]) + "\n")
        print("Done.")
        return attention_values_list

    def t2str(self, t):
        # if self.device=="cpu":
        #     l = [str(val)+"," for val in t.detach().numpy()]
        # else:
        #     l = [str(val)+"," for val in t.cpu().detach().numpy()]
        # result = "".join(l)
        # result = result[:-1]
        if self.device=="cpu":
            result = str(t.detach().numpy())
        else:
            result = str(t.cpu().detach().numpy())
        return result


class Visualizer:
    def __init__(self, input_tensor, output_tensor, input_padding_flags, output_padding_flags, sentence_lists, attention_value_lists, device=torch.device("cpu")) -> None:
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.input_padding_flags = input_padding_flags
        self.output_padding_flags = output_padding_flags
        self.sentence_lists = sentence_lists
        self.attention_value_lists = attention_value_lists
        self.device = device

    def get_clusters(self, eps, min_samples):
        array_list = [[],[]]
        flag_list = [[],[]]
        attention_value_list = [[],[]]

        for tensor in [self.input_tensor, self.input_padding_flags, self.output_tensor, self.output_padding_flags]:
            if tensor.is_cuda:
                tensor.cpu()

        array_list[0] = self.input_tensor.detach().cpu().numpy()
        flag_list[0] = self.input_padding_flags.detach().cpu().numpy()
        attention_value_list[0] = self.attention_value_lists[0].detach().cpu().numpy()
        array_list[1] = self.output_tensor.detach().cpu().numpy()
        flag_list[1] = self.output_padding_flags.detach().cpu().numpy()
        attention_value_list[1] = self.attention_value_lists[1].detach().cpu().numpy()

        sentences_list = []
        cluster_num_list = []
        cluster_id_max_list = []

        for i in range(2):
            sentences, data = self.find_description_embeddings(array_list, flag_list, attention_value_list, i)
            reducer = UMAP(n_neighbors=3, init='random', random_state=0)
            projected_data_2d = reducer.fit_transform(data)
            pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(projected_data_2d)
            sentences_list.append(sentences)
            cluster_num_list.append(list(pred))
            cluster_id_max_list.append(np.max(pred))

            for sentence, cluster in zip(sentences, pred):
                print(sentence + " :> " + str(i) + ":" + str(cluster))

            plt.title(f'Column-{i}')
            plt.scatter(projected_data_2d[:, 0], projected_data_2d[:, 1], c=pred, alpha=0.9)
            plt.colorbar()
            plt.savefig(f'Column-{i}.png')
            plt.clf()

        with open("sentence_file.csv", "w") as f:
            writer = csv.writer(f)
            for s1, c1, s2, c2 in zip(sentences_list[0], cluster_num_list[0], sentences_list[1], cluster_num_list[1]):
                writer.writerow([c1, s1, c2, s2])

        print("done")
        return cluster_num_list, sentences_list, cluster_id_max_list

    def find_description_embeddings(self, emb_array_list, flag_list, attention_value_list, i):
        data_list = []
        sentences = []
        for embed_array, flags, sentence_pair, attentions in zip(
            emb_array_list[i], flag_list[i], self.sentence_lists, attention_value_list[i]
        ):
            attention = flags * attentions
            att_emb = attention @ embed_array
            sentence = sentence_pair[i]
            data_list.append(np.expand_dims(att_emb, axis=0))
            sentences.append(sentence[attention.argmax()])
        desc_embedding_data = np.concatenate(data_list, axis=0)
        print(f"# of sentences({i}): {len(sentences)}")
        return sentences, desc_embedding_data

    def get_bigraph(self, eps, min_samples, filename_suffix="default"):
        cluster_num_list, sentences_list, cluster_id_max_list = self.get_clusters(eps, min_samples)

        for i, max_cluster_id in enumerate(cluster_id_max_list):
            print(f"Column-{i} のクラスタ数: {max_cluster_id + 1}")

        bigraph_net = nx.Graph()
        edge_list = []
        alt_edge_list = []

        for l, r in zip(cluster_num_list[0], cluster_num_list[1]):
            edge_list.append((f"0:{l}", f"1:{r}"))
            alt_edge_list.append(f"0:{l}/1:{r}")

        edge_counter = Counter(alt_edge_list)
        pos = {}

        for k in range(2):
            bigraph_net.add_nodes_from([f"{k}:{i}" for i in range(-1, cluster_id_max_list[k] + 1)], bipartite=k)
            pos.update({f"{k}:{i}": (k + 1, i) for i in range(-1, cluster_id_max_list[k] + 1)})

        for edge in set(edge_list):
            alt_edge = f"{edge[0]}/{edge[1]}"
            bigraph_net.add_edge(edge[0], edge[1], weight=edge_counter[alt_edge])

        edge_weights = [bigraph_net[u][v]["weight"] for u, v in bigraph_net.edges]
        mag = 500
        node_degree = [mag * bigraph_net.degree(weight="weight")[i] for i in bigraph_net.nodes]

        pos_spring = nx.spring_layout(bigraph_net, weight="weight", seed=4)
        for k in range(2):
            for i in range(-1, cluster_id_max_list[k] + 1):
                node = f"{k}:{i}"
                pos[node] = (k + 1, pos_spring[node][1])

        nx.draw(bigraph_net, pos, node_color="red", alpha=0.3, with_labels=True,
                node_size=node_degree, width=edge_weights)

        output_filename = f"bigraph_{filename_suffix}.png"
        plt.savefig(output_filename)
        print(f"Saved bigraph as {output_filename}")
        plt.show()
