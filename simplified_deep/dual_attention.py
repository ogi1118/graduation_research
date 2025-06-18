import bert_model_loader as bml
import sentence_encoder as se
import data_reader as dr
import torch
import torch.nn as nn
from torch import optim


class Attention(nn.Module):
    def __init__(self, flag_tensor, input_shape, hidden_dim=50) -> None:
        super().__init__()
        # input_shape[-2]: # of sentences
        # input_shape[-1]: dim of embedding vec
        input_dim = input_shape[-2]*input_shape[-1]
        out_dim = input_shape[-2]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.act_func = nn.ReLU()
        self.soft_max = nn.Softmax(dim=-1)
        self.one_flags = flag_tensor
        self.zero_flags = torch.ones_like(flag_tensor) - flag_tensor


    def forward(self, x):
        x = x.flatten(-2,-1)
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = self.one_flags * self.act_func(self.fc4(x)) + self.zero_flags * (-50.0)
        x = self.soft_max(x)
        return x


class AttentionEncoder(nn.Module):
    def __init__(self, flag_tensor, input_shape, output_dim, hidden_dim=50) -> None:
        super().__init__()
        # input_shape[-2]: # of sentences
        # input_shape[-1]: dim of embedding vec
        input_dim = input_shape[-1]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.act_func = nn.ReLU()
        self.attention = Attention(flag_tensor, input_shape, hidden_dim=hidden_dim)

    def forward(self, x):
        alpha = self.attention(x).unsqueeze(-2)
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = torch.bmm(alpha, self.fc4(x)).squeeze(-2)
        return x
    
    def get_attention(self, x):
        return self.attention(x)


class Associator(nn.Module):
    def __init__(self, tensor_dim, hidden_dim=50) -> None:
        super().__init__()

        self.fc1 = nn.Linear(tensor_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, tensor_dim)
        self.act_func = nn.ReLU()

    def forward(self, x):
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = self.fc4(x)
        return x


class DualAttention:
    def __init__(self, encoded_vector_dim) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.read_flag = False
        self.encoded_vector_dim = encoded_vector_dim
        self.sentence_lists = None

    def which_device(self):
        return self.device

    def get_sentence_lists(self):
        return self.sentence_lists

    def read(self, max_sentence_number=20, file_name=None, col_names=None):
        # ファイル名とcolumn名が指定されているか確認
        if file_name is None or col_names is None:
            raise ValueError("file_nameとcol_namesを指定してください。")
        else:
            # SentenceConverterを使用してデータを変換
            sc = dr.SentenceConverter(file_name, col_names)
            tensor_group_list = sc.convert(max_sentence_number)

        # 各columnのテンソルとunpaddingフラグを格納するリストを初期化
        tensors = []
        unpadding_flags = []
        for tensor_group in tensor_group_list:
            for col_name in col_names:
                # columnごとのテンソルとフラグをリストに追加
                tensors.append(tensor_group.get(col_name))
                unpadding_flags.append(tensor_group.get_unpadding_flags(col_name))

        self.read_flag = True
        return tensors, unpadding_flags

    def train(self, tensors, padding_flags, epochs=10):
        # 各columnのテンソルに対してエンコーダを作成
        encoders = []
        for tensor, flags in zip(tensors, padding_flags):
            encoder = AttentionEncoder(flags, tensor.shape, self.encoded_vector_dim)
            encoder = encoder.to(self.device)
            encoders.append(encoder)

        # Associatorを作成
        assoc = Associator(self.encoded_vector_dim)
        assoc = assoc.to(self.device)

        # Optimizerを設定
        optimizer = optim.Adam(
            [param for encoder in encoders for param in encoder.parameters()] + list(assoc.parameters()),
            lr=0.0001
        )
        criterion = nn.MSELoss()

        # エポックごとのトレーニング処理
        for epoch in range(epochs):
            # トレーニングロジックをここに記述
            pass

        # モデルを評価モードに設定
        for encoder in encoders:
            encoder.eval()
        assoc.eval()

        return encoders, assoc



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    encoded_vector_dim = 20 
    da = DualAttention(encoded_vector_dim)
    # input_tensor, input_padding_flags, output_tensor, output_padding_flags = da.read(doc)

    # in_encoder = AttentionEncoder(input_padding_flags, input_tensor.shape, encoded_vector_dim)
    # in_encoder = in_encoder.to(device)
    # in_encoded_data = in_encoder(input_tensor)

    # assoc = Associator(encoded_vector_dim)
    # assoc = assoc.to(device)
    # expected_out = assoc(in_encoded_data)

    # out_encoder = AttentionEncoder(output_padding_flags, output_tensor.shape, encoded_vector_dim)
    # out_encoder = out_encoder.to(device)
    # out_encoded_data = out_encoder(output_tensor)

    # print(expected_out)
    # print(out_encoded_data)
    da.train(doc)





