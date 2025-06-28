import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random

# 経験再生用のメモリ
class ReplayBuffer:
    """経験再生用のバッファークラス"""
    def __init__(self, size_max=5000, batch_size=64):
        # バッファーの初期化
        self.buffer = deque(maxlen=size_max)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        # idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        # return [self.buffer[ii] for ii in idx]
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# アクターネットワーク
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'nn_actor_td3.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_state_dict()
        self.to(self.device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a

    def save(self):
        """モデルのパラメータを保存"""
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self, strict=True, assign=False):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device), strict=strict)
            print('...actor network loaded...')

# クリティックネットワーク（Q関数）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'nn_critic_td3.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_state_dict()
        self.to(self.device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1 forward
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2 forward
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    # Q1のみ返す関数（ターゲットネットワーク用など）
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def save(self):
        """モデルのパラメータを保存"""
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self, strict=True):
        """モデルのパラメータをロード"""
        if os.path.isfile(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn), strict=strict)
            print('...critic network loaded...')
        else:
            print('...no critic network found...')
