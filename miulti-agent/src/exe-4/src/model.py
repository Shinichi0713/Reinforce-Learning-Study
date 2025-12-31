import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class GRU_Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.action_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs, h):
        # obs: (batch, seq_len, obs_dim), h: (1, batch, hidden_dim)
        x = self.fc(obs)
        x, h_next = self.gru(x, h)
        probs = torch.softmax(self.action_head(x), dim=-1)
        return Categorical(probs), h_next

class GRU_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            init_layer(nn.Linear(state_dim, hidden_dim)), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.v_head = init_layer(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, state, h):
        x = self.fc(state)
        x, h_next = self.gru(x, h)
        return self.v_head(x), h_next