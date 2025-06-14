
import torch
import torch.nn as nn

# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, embedded_inputs, input_lengths):
        # 可変長のシーケンスをパックしてRNNに入力
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=True)
        enc_out, (h, c) = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        return enc_out, (h, c)

# デコーダの状態とエンコーダの出力よりどの単語に注目すべきかの評価
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decorder_state, enc_out, mask):
        # エンコーダ出力の変換
        encoder_transformer = self.w1(enc_out)
        # デコーダ状態の変換
        decoder_transformer = self.w2(decorder_state).unsqueeze(1)
        # アテンションスコアの計算
        u_i = self.vt(torch.tanh(encoder_transformer + decoder_transformer)).squeeze(-1)
        # log-softmaxはPyTorchや多くのライブラリで数値的に安定した方法で実装されています
        log_score = masked_log_softmax(u_i, mask, dim=-1)
        return log_score

class PointerNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(PointerNet, self).__init__()
        self.hidden_size = hidden_dim
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.encoder = Encoder(embedding_dim, hidden_dim, num_layers=1)
        self.decoder = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.attention = Attention(hidden_dim)

        self.__init_nn__()

    def __init_nn__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_seq, input_lengths):
        # 入力シーケンスサイズ
        batch_size, seq_len, _ = input_seq.size()

        embedded = self.embedding(input_seq)
        enc_out, (encoder_h_n, encoder_c_n) = self.encoder(embedded, input_lengths)

        # 双方向
        enc_out = enc_out[:, :, :self.hidden_size] + enc_out[:, :, self.hidden_size:]
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        
