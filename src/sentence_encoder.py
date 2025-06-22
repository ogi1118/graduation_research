import torch
# import torch.nn.functional as F
from bert_model_loader import BertLoader
import numpy as np
from sentence_group import SentenceGroupList, SentenceGroup, TensorGroup

class SentenceEncoder:
    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.bert = BertLoader(device)
        
    def encode(self, texts):
        cls = self.bert.get_CLS(texts)
        return cls

    def doc_encode(self, doc, max_sentence_number=4):
        doc_embedding_vectors_array = []
        doc_unpadding_flags_array = []
        for texts_array in doc:
            embedding_vectors_array = []
            # max_sentence_number = np.max(np.array([len(texts) for texts in texts_array]))
            unpadding_flags_array = [[1 if i<len(texts) else 0 for i in range(max_sentence_number)] for texts in texts_array]
            for texts, flags in zip(texts_array, unpadding_flags_array):
                cls = self.bert.get_CLS(texts, flags)
                embedding_vectors_array.append(cls.unsqueeze(dim=0))
            doc_embedding_vectors_array.append(torch.cat(embedding_vectors_array).unsqueeze(dim=0))
            doc_unpadding_flags_array.append(torch.tensor(unpadding_flags_array).unsqueeze(dim=0))
        embedding_vectors = torch.cat(doc_embedding_vectors_array)
        unpadding_flags = torch.cat(doc_unpadding_flags_array)
        return embedding_vectors, unpadding_flags

    def encode_sentences(self, sentence_group_list, max_sentence_number=4):
        tensor_group_list = []
        column_names = sentence_group_list.get_column_names()

        for sentence_group in sentence_group_list.get_all():
            sentence_dict = sentence_group.get_sentences_dict()
            tensor_group = TensorGroup(sentence_dict)
            for column_name in column_names:
                texts = sentence_group.get(column_name)
                flags = [1 if i<len(texts) else 0 for i in range(max_sentence_number)]
                cls = self.bert.get_CLS(texts, flags)
                cls = cls.unsqueeze(dim=0)
                tensor_group.set(column_name,cls)
                tensor_group.set_unpadding_flags(column_name, torch.tensor(flags).unsqueeze(dim=0))
            tensor_group_list.append(tensor_group)
        return tensor_group_list

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [
        "We are curious about the background and what are the work values of IHI?",
        "Is there any new information added to this session with Surabaya compared to last week's meeting?",
        "We want to know why engineers want to work for IHI."
    ]

    doc = [
        [
            [
                "We are curious about the background and what are the work values of IHI?",
                "Is there any new information added to this session with Surabaya compared to last week's meeting?",
                "We want to know why engineers want to work for IHI."
            ],
            [
                "At FlexiRoam, we value flexibility as much as you do.",
                "That's why our eSIM plans are designed to adapt to your travel needs, giving you the freedom to roam wherever."
            ], 
        ],
        [
            [
                "We are curious about the background and what are the work values of IHI?",
                "Is there any new information added to this session with Surabaya compared to last week's meeting?",
                "We want to know why engineers want to work for IHI."
            ],
            [
                "At FlexiRoam, we value flexibility as much as you do.",
                "That's why our eSIM plans are designed to adapt to your travel needs, giving you the freedom to roam wherever."
            ], 
        ]
    ]

    se = SentenceEncoder()
    embedding_tensor, unpadding_flags = se.doc_encode(doc, max_sentence_number=4)
    print(embedding_tensor.shape)
    print(unpadding_flags.shape)
    print(embedding_tensor)
    print(unpadding_flags)