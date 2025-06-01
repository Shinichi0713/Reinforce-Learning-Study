# DQNのモデル
import torch
import torch.nn as nn
import os

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_nn = os.path.join(os.path.dirname(__file__), 'dqn_model.pth')
        self.__load_state_dict()
        
        self.to(self.device)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = self.fc(x)
        return self.softmax(x)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device))
            print(f"Model loaded from {self.path_nn}")
        else:
            print(f"No model found at {self.path_nn}, starting with random weights.")

    def save_model(self):
        torch.save(self.state_dict(), self.path_nn)
        print(f"Model saved to {self.path_nn}")