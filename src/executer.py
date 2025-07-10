import os
from config_loader import config
import json
from dual_attention import DualAttention
from visualizer import Visualizer, CombinedVisualizer
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


if __name__ == "__main__":
    file_name = config["file_name"]
    col_names = config["col_names"]
    outputs_dir = config["outputs_dir"]
    tmp_dir = config["tmp_dir"]
    outputs_graphs_dir = config["outputs_graphs_dir"]
    outputs_csvs_dir = config["outputs_csvs_dir"]

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(outputs_graphs_dir, exist_ok=True)
    os.makedirs(outputs_csvs_dir, exist_ok=True)
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
        output_file_name = outputs_csvs_dir + "/[" + \
            col_name_from + "]2[" + col_name_to + "].csv"
        attention_value_lists = ae.evaluate(output_file_name)

        vis = Visualizer(input_tensor, output_tensor, input_padding_flags, output_padding_flags,
                         sentence_lists, attention_value_lists, device=executer.which_device(), col_name_from=col_name_from, col_name_to=col_name_to)
        vis.save_cluster_info(eps=0.7, min_samples=3)
        # vis.get_bigraph(0.7, 3)

    # cluster_infos = CombinedVisualizer.load_cluster_infos()
    merged_infos = CombinedVisualizer.merge_and_recluster(
        eps=0.7, min_samples=3)

    CombinedVisualizer.draw_merged_bigraph(merged_infos)
