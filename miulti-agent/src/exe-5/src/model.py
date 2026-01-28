import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# --- Actor: 個別の観測から行動を決定 (パラメータ共有) ---
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 入力: obs + agent_id
        self.fc1 = nn.Linear(obs_dim + 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        probs = F.softmax(self.fc2(x), dim=-1)
        return Categorical(probs), h

# --- Critic: 全員の観測を結合して価値を推定 (集中型) ---
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        return self.fc2(x), h

# --- Memory: シーケンスデータを保存 ---
class MAPPOMemory:
    def __init__(self):
        self.obs, self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], [], []
        self.h_actors, self.h_critics = [], []

    def store(self, obs, state, action, log_prob, reward, done):
        self.obs.append(obs); self.states.append(state); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward); self.dones.append(done)

    def clear(self):
        self.obs, self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], [], []
        self.h_actors, self.h_critics = [] , []