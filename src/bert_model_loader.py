import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BertLoader:
    def __init__(self, device) -> None:
        sbert_bin = "sentence-transformers/all-distilroberta-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(sbert_bin)
        self.model = AutoModel.from_pretrained(sbert_bin)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    def get_models(self):
        return self.tokenizer, self.model

    def get_encodings(self, texts):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encodings = encodings.to(self.device)
        return encodings

    def get_CLS(self, texts, flags=None):
        encodings = self.get_encodings(texts)
        with torch.no_grad():
            embeds = self.model(**encodings)
        token_embeds = embeds[0]
        cls = token_embeds[:, 0, :]
        if flags is not None:
            vector_dim = cls.shape[1]
            zero_flag_num = flags.count(0)
            zero_vectors = torch.zeros(
                (zero_flag_num, vector_dim)).to(self.device)
            cls = torch.cat((cls, zero_vectors), dim=0)
        return cls

    def get_avepool(self, texts):
        encodings = self.get_encodings(texts)
        with torch.no_grad():
            embeds = self.model(**encodings)
        token_embeds = embeds[0]
        attention_mask = encodings["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeds.size()).float()
        sentence_embeds = torch.sum(token_embeds * input_mask_expanded,
                                    dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=0.1)
        return sentence_embeds


def get_cos_sim(cls):
    normalized_cls = F.normalize(cls, p=2, dim=1)
    cls_similarity = normalized_cls.matmul(normalized_cls.T)
    print(cls_similarity)
    return cls_similarity


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [
        "We are curious about the background and what are the work values of IHI?",
        "Is there any new information added to this session with Surabaya compared to last week's meeting?",
        "We want to know why engineers want to work for IHI."
    ]
    bert = BertLoader(device)
    tokenizer, model = bert.get_models()
    encodings = bert.get_encodings(texts)
    attention_mask = encodings["attention_mask"]

    with torch.no_grad():
        embeds = model(**encodings)
    token_embeds = embeds[0]
    s_tag_embeds = embeds[1]

    for tokens in encodings["input_ids"]:
        print(tokenizer.convert_ids_to_tokens(tokens))
    print(attention_mask)
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeds.size()).float()
    sentence_embeds = torch.sum(token_embeds * input_mask_expanded,
                                dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=0.1)

    print(token_embeds.shape)
    print(s_tag_embeds.shape)
    cls = token_embeds[:, 0, :]
    print(token_embeds[:, 0, :])  # CLS
    print(s_tag_embeds)
    print(sentence_embeds)  # average pooling

    normalized_cls = F.normalize(cls, p=2, dim=1)
    cls_similarity = normalized_cls.matmul(normalized_cls.T)
    print(cls_similarity)

    normalized_stag = F.normalize(s_tag_embeds, p=2, dim=1)
    stag_similarity = normalized_stag.matmul(normalized_stag.T)
    print(stag_similarity)

    sbert_stag = F.normalize(sentence_embeds, p=2, dim=1)
    sbert_similarity = sbert_stag.matmul(sbert_stag.T)
    print(sbert_similarity)
