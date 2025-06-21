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
    def __init__(self, state_size=4, action_size=2, hidden_size=100):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = f"{dir_current}/nn_parameter_actor.pth"

        self.__load_nn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = self.model(x)
        x = torch.softmax(x, dim=-1)
        return x

    def save_nn(self):
        self.cpu()
        torch.save(self.model.state_dict(), self.path_nn)
        self.to(self.device)
    
    def __load_nn(self):
        if os.path.exists(self.path_nn):
            self.model.load_state_dict(torch.load(self.path_nn))
        else:
            print(f"Model file {self.path_nn} does not exist. Skipping load.")


# 行動価値を評価するためのCriticクラス
class Critic(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=100):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = f"{dir_current}/nn_parameter_critic.pth"

        self.__load_nn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, state, action):
        x = torch.cat([torch.tensor(state, dtype=torch.float32).to(self.device), 
                       torch.tensor(action, dtype=torch.float32).to(self.device)], dim=-1)
        return self.model(x)

    def save_nn(self):
        self.cpu()
        torch.save(self.model.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_nn(self):
        if os.path.exists(self.path_nn):
            self.model.load_state_dict(torch.load(self.path_nn))
        else:
            print(f"Model file {self.path_nn} does not exist. Skipping load.")

