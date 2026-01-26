class SentenceGroupList:
    def __init__(self) -> None:
        self.sentence_group_list = []
        self.column_names = None

    def add(self, sg):
        self.sentence_group_list.append(sg)

    def get(self, i):
        return self.sentence_group_list[i]
    
    def get_all(self):
        return self.sentence_group_list
    
    def print(self):
        for sg in self.sentence_group_list:
            sg.print()
            print("----------")

    def set_column_names(self, column_names):
        self.column_names = column_names

    def get_column_names(self):
        return self.column_names
    
class SentenceGroup:
    def __init__(self) -> None:
        self.sentences_array = []
        self.sentences_dict = {}

    def set(self, key, sentences):
        if key in self.sentences_dict:
            self.sentences_array[self.sentences_dict[key]] = sentences
        else:
            self.sentences_array.append(sentences)
            self.sentences_dict[key] = len(self.sentences_array) - 1

    def get(self, key):
        if key in self.sentences_dict:
            return self.sentences_array[self.sentences_dict[key]]
        else:
            raise ValueError(f"No such key!:{key}")
        
    def get_by_id(self, id):
        if id >= len(self.sentences_array):
            raise ValueError(f"No such id!:{id}")
        else:
            return self.sentence_array[id]
        
    def print(self):
        for key in self.sentences_dict:
            print(key + ":" + self.sentences_array[self.sentences_dict[key]][0]+"...")

    def get_sentences_dict(self):
        return self.sentences_dict
    
    def get_all_sentences(self):
        return self.sentences_array

class TensorGroup:
    def __init__(self, sentences_dict) -> None:
        self.sentences_dict = sentences_dict
        self.tensor_list = [None for _ in range(len(sentences_dict))]
        self.unpadding_flags_list = [None for _ in range(len(sentences_dict))]

    def set(self, key, t):
        self.tensor_list[self.sentences_dict[key]] = t

    def get(self, key):
        if key in self.sentences_dict:
            return self.tensor_list[self.sentences_dict[key]]
        else:
            raise ValueError(f"No such key!:{key}")
        
    def set_unpadding_flags(self, key, unpadding_flags_tensor):
        self.unpadding_flags_list[self.sentences_dict[key]] = unpadding_flags_tensor

    def get_unpadding_flags(self, key):
        return self.unpadding_flags_list[self.sentences_dict[key]]    


if __name__ == "__main__":
    sg = SentenceGroup()
    sg.set("Q1", ["ABC","AAA"]) # insert
    sg.set("Q2", ["DEF"]) # insert
    sg.set("Q1", ["GHI","DEF"]) # update
    sg.print()
    