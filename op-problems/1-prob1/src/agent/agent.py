
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

def masked_max(vector: torch.Tensor,
			   mask: torch.Tensor,
			   dim: int,
			   keepdim: bool = False,
			   min_val: float = -1e7):
    """
    To calculate max along certain dimensions on masked values
    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings
    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

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
        self.decoding_rnn = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
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
        encoder_outputs, (encoder_h_n, encoder_c_n) = self.encoder(embedded, input_lengths)

        # 双方向
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())

        range_tensor = torch.arange(seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(batch_size, seq_len, seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, seq_len, seq_len)
        
        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(seq_len):
            sub_mask = mask_tensor[:, i, :].float()

            # デコーダの状態を更新
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
            # 次の状態
            decoder_hidden = (h_i, c_i)
            # アテンションスコアを計算
            log_pointer_score = self.attention(h_i, encoder_outputs, sub_mask)
            # スコアを保存
            pointer_log_scores.append(log_pointer_score)

            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)
            # 最大のポインタを取得
            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

            # 次の入力として最大値のインデックスを使用
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        # スコアを結合
        pointer_log_scores = torch.stack(pointer_log_scores, dim=1)
        # ポインタのインデックスを結合
        pointer_argmaxs = torch.stack(pointer_argmaxs, dim=1)

        return pointer_log_scores, pointer_argmaxs, mask_tensor
