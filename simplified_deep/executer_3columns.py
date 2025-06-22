# executer_3columns.py

from dual_attention import DualAttention
from executer import AttentionEvaluator, Visualizer

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
            col_name_to=col_name_to,
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


def run_all_combinations(file_name, col_names, max_sentence_number=20, epochs=100):
    for i in range(len(col_names)):
        for j in range(i, len(col_names)):
            if i < j:
                continue

            col_from = col_names[i]
            col_to = col_names[j]
            print(f"\n===== Processing {col_from} → {col_to} =====")

            executer = Executer(file_name)
            in_encoder, out_encoder, assoc, input_tensor, output_tensor, input_padding_flags, output_padding_flags = executer.train(
                col_from, col_to, max_sentence_number=max_sentence_number, epochs=epochs
            )

            sentence_lists = executer.get_sentence_lists()
            ae = AttentionEvaluator(
                in_encoder, out_encoder,
                input_tensor, output_tensor,
                sentence_lists,
                device=executer.which_device()
            )

            output_file_name = f"./outputs/[{col_from}]2[{col_to}].csv"
            attention_value_lists = ae.evaluate(output_file_name)

            vis = Visualizer(
                input_tensor, output_tensor,
                input_padding_flags, output_padding_flags,
                sentence_lists, attention_value_lists,
                device=executer.which_device()
            )
            vis.get_bigraph(0.7, 3, f"{col_from}_2_{col_to}")



if __name__ == "__main__":
    col_names = [
        "interesting_facts_or_information",
        "identified_problems_or_needs",
        "data_or_information_to_be_further_explored",
    ]
    file_name = "../dataset20240126.csv"
    run_all_combinations(file_name, col_names, max_sentence_number=20, epochs=100)
