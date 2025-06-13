
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        enc_out, (h, c) = self.rnn(x)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        return enc_out, (h, c)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, enc_out):
        attn_weights = torch.tanh(self.w1(enc_out))
        attn_weights = torch.matmul(attn_weights, self.w2(enc_out).transpose(1, 2))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * enc_out, dim=1)
        return context, attn_weights


