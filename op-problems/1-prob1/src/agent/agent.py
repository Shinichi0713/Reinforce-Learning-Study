
import torch
import torch.nn as nn

# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
	"""
	``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
	``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
	broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	In the case that the input vector is completely masked, the return value of this function is
	arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
	of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
	that we deal with this case relies on having single-precision floats; mixing half-precision
	floats with fully-masked vectors will likely give you ``nans``.
	If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
	lower), the way we handle masking here could mess you up.  But if you've got logit values that
	extreme, you've got bigger problems than this.
	"""
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
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        enc_out, (h, c) = self.rnn(x)
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



