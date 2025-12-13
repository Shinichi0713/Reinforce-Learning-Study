import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        
        # 観測を入力とする線形層
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        # 履歴を扱うGRUセル
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        # 隠れ状態から行動のQ値を出力する線形層
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        
    def init_hidden(self):
        # 隠れ状態をゼロで初期化
        return self.fc1.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        # inputs: 局所的な観測 (obs_i)
        # hidden_state: 前ステップの隠れ状態 (h_t-1)
        
        x = F.relu(self.fc1(inputs))
        # GRUによる隠れ状態の更新
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        # Q値の出力
        q = self.fc2(h)
        return q, h # Q値と新しい隠れ状態を返す

#