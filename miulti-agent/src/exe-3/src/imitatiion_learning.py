import torch
import torch.nn as nn
from torch.distributions import Categorical

# Actor: 自分の観測から行動の確率を出力
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, obs):
        return Categorical(self.net(obs))

# Critic: 全員の状態(State)から価値(V)を算出
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        return self.net(state)
    
# 1. データの収集 (Rollout)
# 複数ステップ分、(obs, state, action, log_prob, reward) を貯める

# 2. Criticの更新
# 全員の状態(State)を使って、TD誤差を最小化
v_targets = rewards + gamma * next_values
critic_loss = F.mse_loss(critic(states), v_targets.detach())

# 3. Actorの更新 (PPOのコア)
# アドバンテージ A = v_target - v_current を計算
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
actor_loss = -torch.min(surr1, surr2).mean()