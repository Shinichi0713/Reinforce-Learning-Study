# エージェント
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
import os
import numpy as np

# 経験再生用のメモリ
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)


# 行動するActor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
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

