import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        # ハイパーネットワーク1 (重み w1 を生成)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * self.embed_dim)
        # ハイパーネットワーク2 (重み w2 を生成)
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim * 1)

        # バイアス項
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.v_head = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: 各エージェントのQ値 [batch_size, n_agents]
        states: グローバル状態ベクトル [batch_size, state_dim]
        """
        batch_size = agent_qs.size(0)
        
        # ここに実装を記述してください
        # 1. agent_qs を [batch_size, 1, n_agents] にリシェイプ
        # 2. states から w1, b1, w2, b2(v_head) を生成
        # 3. w1, w2 に絶対値を適用して正値を保証
        # 4. 行列演算を行い Q_tot を算出
        
        return q_tot