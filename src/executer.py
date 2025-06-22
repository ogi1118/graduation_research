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
  def __init__(self, file_name, col_name_from, col_name_to) -> None:
    self.encoded_vector_dim = 20 
    self.da = DualAttention(self.encoded_vector_dim)
    self.device = self.da.which_device()
    self.file_name, self.col_name_from, self.col_name_to = file_name, col_name_from, col_name_to
    self.sentence_lists = None

  def train(self, max_sentence_number=20, epochs=100):
    input_tensor, input_padding_flags, output_tensor, output_padding_flags = self.da.read(max_sentence_number=max_sentence_number, file_name=self.file_name, 
        col_name_from=self.col_name_from, col_name_to=self.col_name_to)
    in_encoder, out_encoder, assoc = self.da.train(input_tensor, input_padding_flags, output_tensor, output_padding_flags, epochs=epochs)
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
        f.write(in_or_out[i]+"\n")
        for j in range(len(self.sentence_lists)):
          sentence_list = self.sentence_lists[j][i]
          f.write("..."+str(j)+"................."+"\n")
          for k in range(len(sentence_list)):
            f.write(sentence_list[k]+":"+self.t2str(attention_values[j,k])+"\n" )
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
    
    if self.input_tensor.is_cuda:
      self.input_tensor = self.input_tensor.cpu()
      self.input_padding_flags = self.input_padding_flags.cpu()
      self.output_tensor = self.output_tensor.cpu()
      self.output_padding_flags = self.output_padding_flags.cpu()
      self.attention_value_lists[0] = self.attention_value_lists[0].cpu()
      self.attention_value_lists[1] = self.attention_value_lists[1].cpu()
    array_list[0] = self.input_tensor.detach().numpy()
    flag_list[0] = self.input_padding_flags.detach().numpy()
    attention_value_list[0] = self.attention_value_lists[0].detach().numpy()
    array_list[1] = self.output_tensor.detach().numpy()
    flag_list[1] = self.output_padding_flags.detach().numpy()
    attention_value_list[1] = self.attention_value_lists[1].detach().numpy()
    
    sentences_list = []
    cluster_num_list = []
    cluster_id_max_list = []
    for i in range(2):
      sentences, data = self.find_description_embeddings(array_list, flag_list, attention_value_list, i)
      reducer = UMAP(n_neighbors=3, init='random', random_state=0) #5
      projected_data_2d = reducer.fit_transform(data)
      pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(projected_data_2d)
      sentences_list.append(sentences)
      cluster_num_list.append(list(pred))
      cluster_id_max_list.append(np.max(pred))
      for sentence, cluster in zip(sentences, pred):
        print(sentence+" :> "+str(i)+":"+str(cluster))
      plt.title('Column-{}'.format(str(i)))
      plt.scatter(projected_data_2d[:, 0], projected_data_2d[:, 1], c=pred, alpha=0.9)
      plt.colorbar()
      plt.savefig('../outputs/Column-{}.png'.format(str(i)))
      plt.clf()
    with open("sentence_file.csv", "w") as f:
      writer = csv.writer(f)
      for sentence_from, cluster_from, sentence_to, cluster_to in zip(sentences_list[0], cluster_num_list[0], sentences_list[1], cluster_num_list[1]):
        writer.writerow([cluster_from, sentence_from , cluster_to,  sentence_to])

    print("done")
    return cluster_num_list, sentences_list, cluster_id_max_list

  def find_description_embeddings(self, emb_array_list, flag_list, attention_value_list, i):
    data_list = []
    sentences = []
    for embed_array, flags, sentence_list, attentions in zip(emb_array_list[i], flag_list[i], self.sentence_lists, attention_value_list[i]):
      attention = flags * attentions
      att_emb = attention @ embed_array
      sentence = sentence_list[i] 
      data_list.append(np.expand_dims(att_emb,axis=0))
      sentences.append(sentence[attention.argmax()])
    desc_embedding_data = np.concatenate(data_list, axis=0)
    print("# of sentences({}): {}".format(i,len(sentences)))
    return sentences,desc_embedding_data

  def get_bigraph(self, eps, min_samples):
    cluster_num_list, sentences_list, cluster_id_max_list = self.get_clusters(eps, min_samples)
    bigraph_net = nx.Graph()
    edge_list = []
    alt_edge_list = []
    for l, r in zip(cluster_num_list[0], cluster_num_list[1]):
      edge_list.append(("0:" + str(l), "1:" + str(r)))
      alt_edge_list.append("0:" + str(l) + "/" + "1:" + str(r))
    edge_counter = Counter(alt_edge_list)
    pos = dict()
    for k in range(2):
      bigraph_net.add_nodes_from([str(k) + ":" + str(i) for i in range(-1, cluster_id_max_list[k] + 1)], bipartite=k)
      pos.update((str(k) + ":" + str(i), (k + 1, i)) for i in range(-1, cluster_id_max_list[k] + 1))
    for edge in list(set(edge_list)):
      alt_edge = edge[0] + "/" + edge[1]
      bigraph_net.add_edge(edge[0], edge[1], weight=edge_counter[alt_edge])
    edge_weights = [bigraph_net[u][v]["weight"] for u, v in bigraph_net.edges]
    mag = 500
    node_degree = [mag * bigraph_net.degree(weight="weight")[i] for i in bigraph_net.nodes]
    pos_spring = nx.spring_layout(bigraph_net, weight="weight", seed=4)
    for k in range(2):
      pos.update((str(k) + ":" + str(i), (k + 1, pos_spring[str(k) + ":" + str(i)][1])) for i in range(-1, cluster_id_max_list[k] + 1))
    nx.draw(bigraph_net, pos, node_color="red", alpha=0.3, with_labels=True, node_size=node_degree, width=edge_weights)
    plt.savefig("bigraph.png")
    plt.show()



if __name__ == "__main__":
  file_name = "/home/al22091/graduation_research/dataset20240126.csv"
  col_names = ["curiosity", "interesting_facts_or_information", "identified_problems_or_needs"]
  for i in range(len(col_names)-1):
    for j in range(i+1, len(col_names)):
      if i < j:
        continue
    col_name_from = col_names[i]
    col_name_to = col_names[j]
    print("col_name_from: {}, col_name_to: {}".format(col_name_from, col_name_to))
    executer = Executer(file_name, col_name_from, col_name_to)
    in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(max_sentence_number=20, epochs=1000)
    sentence_lists = executer.get_sentence_lists()
    ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists, device=executer.which_device())
    output_file_name = "../outputs/[" + col_name_from + "]2[" + col_name_to + "].csv"
    attention_value_lists = ae.evaluate(output_file_name)


    vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags, sentence_lists, attention_value_lists, device=executer.which_device())
    vis.get_bigraph(0.7, 3)
  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  # executer = Executer(file_name, col_name_from, col_name_to)
  # in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(max_sentence_number=20, epochs=1000)
  # sentence_lists = executer.get_sentence_lists()
  # ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists, device=executer.which_device())
  # output_file_name = "./outputs/[" + col_name_from + "]2[" + col_name_to + "].csv"
  # attention_value_lists = ae.evaluate(output_file_name)
  # vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags, sentence_lists, attention_value_lists, device=executer.which_device())
  # #vis.get_clusters(0.5, 3)
  # vis.get_bigraph(0.7, 3) #0.5, 3
    
        
