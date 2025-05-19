# エージェントの定義
import torch
import torch.nn as nn
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, "policy_net.pth")
        if os.path.exists(self.path_model):
            print("Loading model from {}".format(self.path_model))
            self.load(self.path_model)

    def forward(self, x):
        x = x.to(self.device)
        return self.fc(x)

    # 行動（action）を選択する処理
    # 確率的方策（ポリシー）に基づいて行動をサンプリング
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        probs = self(state_tensor)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def save(self):
        torch.save(self.state_dict(), self.path_model)

    def load(self):
        self.load_state_dict(torch.load(self.path_model))