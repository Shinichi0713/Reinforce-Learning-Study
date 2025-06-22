# 模倣学習のエージェント
import torch
import torch.nn as nn
from collections import deque
import os


class AgentImitation(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AgentImitation, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.path_nn = f"{os.path.dirname(os.path.abspath(__file__))}/nn_parameter_imitation.pth"
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

