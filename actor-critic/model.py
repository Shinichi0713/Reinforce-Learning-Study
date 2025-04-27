# モデルのstruct
import math
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        self.affine = nn.Linear(4, 128)

        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x):
        
        x = F.relu(self.affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values
    
    # エージェントが行動を選択
    def select_action(self, state, device):

        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state.to(device))

        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), state_value))

        return action.item(), state_value

