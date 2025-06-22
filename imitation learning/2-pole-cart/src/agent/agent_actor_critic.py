# エージェント
import torch
import torch.nn as nn
from collections import deque
import os, random
import numpy as np


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),  # 明示的にnp.array化
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# 行動するActor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.path_nn = f"{os.path.dirname(os.path.abspath(__file__))}/nn_parameter_actor.pth"
        self.__load_nn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)

    def save_nn(self):
        self.cpu()
        torch.save(self.net.state_dict(), self.path_nn)
        self.to(self.device)
    
    def __load_nn(self):
        if os.path.exists(self.path_nn):
            self.net.load_state_dict(torch.load(self.path_nn))
        else:
            print(f"Model file {self.path_nn} does not exist. Skipping load.")


# 行動価値を評価するためのCriticクラス
class Critic(nn.Module):
    def __init__(self, state_dim=4):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = f"{dir_current}/nn_parameter_critic.pth"

        self.__load_nn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x).squeeze(-1)

    def save_nn(self):
        self.cpu()
        torch.save(self.net.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_nn(self):
        if os.path.exists(self.path_nn):
            self.net.load_state_dict(torch.load(self.path_nn))
        else:
            print(f"Model file {self.path_nn} does not exist. Skipping load.")

