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

    def get_sentence_lists(self, column_names):
        # データ処理用のリストを初期化
        doc_list = []  # 各columnの文リストを格納するリスト
        max_sentence_num_list = []  # 各columnの最大文数を格納するリスト

        for column_name in column_names:
            doc = []  # 現在のcolumnの文リストを一時的に格納するリスト
            max_sentence_num = -1  # 現在のcolumnのrow内での最大文数を追跡する変数

            for row in self.df[column_name]:
                # 各rowを処理
                sentences = self.spacy_model(row).sents  # spaCyを使用してrowを文に分割
                sentence_list = [s.text for s in sentences]  # 文をテキストに変換してリストに格納
                doc.append(sentence_list)  # columnの文リストに追加

                # 現在のcolumnの最大文数を更新
                sentence_num = len(sentence_list)
                if sentence_num > max_sentence_num:
                    max_sentence_num = sentence_num

            doc_list.append(doc)  # 処理済みのcolumnの文リストをメインリストに追加
            max_sentence_num_list.append(max_sentence_num)  # columnの最大文数をリストに追加

        # 各columnの文リストと最大文数リストを返す
        return doc_list, max_sentence_num_list

    def get_sentence_groups(self, column_names):
        # 指定されたcolumnsの文リストと最大文数を取得
        doc_list, max_sentence_num_list = self.get_sentence_lists(column_names)
        doc_length_list = [len(doc) for doc in doc_list]  # 各columnのrow数を計算

        # SentenceGroupListを初期化して文をグループ化
        sgl = SentenceGroupList()
        sgl.set_column_names(column_names)  # SentenceGroupListにcolumn名を設定

        for i in range(min(doc_length_list)):  # rowを繰り返し処理し、すべてのcolumnにデータがあるrowのみ処理
            sg = SentenceGroup()  # 現在のrowのSentenceGroupを作成

            for key, doc in zip(column_names, doc_list):
                # 各columnの文をSentenceGroupに追加
                sg.set(key, doc[i])

            sgl.add(sg)  # SentenceGroupをSentenceGroupListに追加

        # SentenceGroupListとすべてのcolumnでの最大文数を返す
        return sgl, max(max_sentence_num_list)

class SentenceConverter:
    def __init__(self, csv_file_name, column_names) -> None:
        sentence_finder = SentenceFinder(csv_file_name)
        self.sentence_group_list, self.max_sentence_num = sentence_finder.get_sentence_groups(column_names)

    def convert(self, max_sentence_number):
        if max_sentence_number < self.max_sentence_num:
            print("Max number of sentences for padding needs to be larger: {}".format(self.max_sentence_num))
            return None
        se = SentenceEncoder()
        tensor_group_list = se.encode_sentences(self.sentence_group_list, max_sentence_number=max_sentence_number)
        return tensor_group_list

    def get_sentence_groups(self):
        return self.sentence_group_list

    def get_sentence_lists(self):
        sentence_lists = []
        for sentence_group in self.sentence_group_list.get_all():
            sentence_lists.append(sentence_group.get_all_sentences())
        return sentence_lists


if __name__ == '__main__':
    sf = SentenceFinder("/home/masaomi/venv/source/gpbl_log/gpbl_log/dataset20240126.csv")
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