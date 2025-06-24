import pandas as pd
import spacy
from sentence_encoder import SentenceEncoder
from sentence_group import SentenceGroup, SentenceGroupList

# pip install -U spacy
# python -m spacy download en_core_web_sm


class SentenceFinder:
    def __init__(self, csv_file_name) -> None:
        self.df = pd.read_csv(csv_file_name, sep=";")
        self.spacy_model = spacy.load("en_core_web_sm")

    def get_sentence_list(self, column_name):
        doc = []
        max_sentence_num = -1
        for row in self.df[column_name]:
            sentences = self.spacy_model(row).sents
            sentence_list = [s.text for s in sentences]  # list(sentences)
            doc.append(sentence_list)
            sentence_num = len(sentence_list)
            if sentence_num > max_sentence_num:
                max_sentence_num = sentence_num
        return doc, max_sentence_num

    def get_sentence_pair_list(self, column_name1, column_name2):
        doc1, max_sentence_num1 = self.get_sentence_list(column_name1)
        doc2, max_sentence_num2 = self.get_sentence_list(column_name2)
        doc = []
        for sentences1, sentences2 in zip(doc1, doc2):
            doc.append([sentences1, sentences2])
        return doc, max(max_sentence_num1, max_sentence_num2)

    def get_sentence_groups(self, column_names):
        doc_list = []
        max_sentence_num_list = []
        doc_length_list = []
        sgl = SentenceGroupList()
        sgl.set_column_names(column_names)
        for column_name in column_names:
            doc, max_sentence_num = self.get_sentence_list(column_name)
            doc_list.append(doc)
            max_sentence_num_list.append(max_sentence_num)
            doc_length_list.append(len(doc))
        for i in range(min(doc_length_list)):  # Rowの数が違う際のmin
            sg = SentenceGroup()
            for key, doc in zip(column_names, doc_list):
                sg.set(key, doc[i])
            sgl.add(sg)
        return sgl, max(max_sentence_num_list)


class SentenceConverter:
    def __init__(self, csv_file_name, column_name1, column_name2) -> None:
        sentence_finder = SentenceFinder(csv_file_name)
        self.sentence_list_pair, self.max_sentence_num = sentence_finder.get_sentence_pair_list(
            column_name1, column_name2)
        self.sentence_lists = [sentence_finder.get_sentence_list(
            column_name1), sentence_finder.get_sentence_list(column_name2)]

    def convert(self, max_sentence_number):
        if max_sentence_number < self.max_sentence_num:
            print("Max number of sentences for padding needs to be larger: {}".format(
                self.max_sentence_num))
            return None, None
        se = SentenceEncoder()
        embedding_tensor, unpadding_flags = se.doc_encode(
            self.sentence_list_pair, max_sentence_number=max_sentence_number)
        return embedding_tensor, unpadding_flags

    def get_sentence_list_pair(self):
        return self.sentence_list_pair

    def get_sentence_lists(self):
        return self.sentence_lists


class SentenceConverter2:
    def __init__(self, csv_file_name, column_names) -> None:
        sentence_finder = SentenceFinder(csv_file_name)
        self.sentence_group_list, self.max_sentence_num = sentence_finder.get_sentence_groups(
            column_names)

    def convert(self, max_sentence_number):
        if max_sentence_number < self.max_sentence_num:
            print("Max number of sentences for padding needs to be larger: {}".format(
                self.max_sentence_num))
            return None
        se = SentenceEncoder()
        tensor_group_list = se.encode_sentences(
            self.sentence_group_list, max_sentence_number=max_sentence_number)
        return tensor_group_list

    def get_sentence_list_pair(self):
        return self.sentence_group_list

    def get_sentence_lists(self):
        sentence_lists = []
        for sentence_group in self.sentence_group_list.get_all():
            sentence_lists.append(sentence_group.get_all_sentences())
        return sentence_lists


if __name__ == '__main__':
    sf = SentenceFinder(
        "/home/masaomi/venv/source/gpbl_log/gpbl_log/dataset20240126.csv")
    # doc, max_sentence_num = sf.get_sentence_list("curiosity")
    # print(doc)
    # print(max_sentence_num)
    # print("#################")
    # doc, max_sentence_num = sf.get_sentence_list("interesting_facts_or_information")
    # print(doc)
    # print(max_sentence_num)
    # print("#################")
    # doc, max_sentence_num = sf.get_sentence_pair_list("curiosity", "interesting_facts_or_information")
    # print(doc)
    # print(max_sentence_num)
    # sgl = sf.get_sentence_group(["curiosity", "interesting_facts_or_information"])
    # sgl.print()
    print("done.")

    # sc = SentenceConverter("/home/masaomi/venv/source/gpbl_log/gpbl_log/dataset20240126.csv", "curiosity", "interesting_facts_or_information")
    # embedding_tensor, unpadding_flags = sc.convert(10) # Error
    # embedding_tensor, unpadding_flags = sc.convert(20) # Success
    # print(embedding_tensor.shape)
    # print(unpadding_flags.shape)
    # print(embedding_tensor)
    # print(unpadding_flags)
