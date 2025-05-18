from data_reader import SentenceConverter
from dual_attention import DualAttention

class Executer:
    def __init__(self, file_name, col_name_from, col_name_to) -> None:
        self.encoded_vector_dim = 20 
        self.da = DualAttention(self.encoded_vector_dim)
        self.file_name, self.col_name_from, self.col_name_to = file_name, col_name_from, col_name_to
        self.sentence_list_pair = None

    def train(self, max_sentence_number=20, epochs=100):
        input_tensor, input_padding_flags, output_tensor, output_padding_flags = self.da.read(None, max_sentence_number=max_sentence_number, appy_executer=True, file_name=self.file_name, 
                col_name_from=self.col_name_from, col_name_to=self.col_name_to)
        in_encoder, out_encoder, assoc = self.da.train(input_tensor, input_padding_flags, output_tensor, output_padding_flags, epochs=epochs)
        self.sentence_lists = self.da.get_sentence_lists()
        return in_encoder, out_encoder, assoc, input_tensor, output_tensor
    
    def get_sentence_lists(self):
        return self.sentence_lists


class AttentionEvaluator:
    def __init__(self, in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists) -> None:
        self.encoders = [in_encoder, out_encoder]
        for encoder in self.encoders:
            encoder.eval()
        self.sentence_tensors = [input_tensor, output_tensor]
        self.sentence_lists = sentence_lists

    def evaluate(self):
        file_name = "out.csv"
        with open(file_name, mode="w") as f:
            for encoder, sentence_tensor, sentence_lists in zip(self.encoders, self.sentence_tensors, self.sentence_lists):
                attention_value_list = encoder.get_attention(sentence_tensor)
                sentence_list = sentence_lists[0]
                print("==========================")
                f.write("==========================\n")
                j = 0
                for sentences, attention_values in zip(sentence_list, attention_value_list):
                    print(".........................")
                    j+=1
                    f.write(str(j)+", .........................\n")
                    for i in range(len(sentences)):
                        print("'"+sentences[i]+"' : "+str(attention_values[i].item()))
                        f.write(str(j)+","+str(i)+","+sentences[i]+", "+str(attention_values[i].item())+"\n")

    def t2str(self, t):
        l = [str(val)+"," for val in t.detach().numpy()]
        result = "".join(l)
        result = result[:-1]
        return result

if __name__ == "__main__":
    file_name = "/home/al22091/graduation_research/dataset20240126.csv"
    col_name_from = "curiosity"
    col_name_to = "interesting_facts_or_information"
    executer = Executer(file_name, col_name_from, col_name_to)
    in_encoder, out_encoder, assoc, input_tensor, output_tensor = executer.train(max_sentence_number=20, epochs=2000)
    sentence_lists = executer.get_sentence_lists()
    ae = AttentionEvaluator(in_encoder, out_encoder, input_tensor, output_tensor, sentence_lists)
    ae.evaluate()
    
        
