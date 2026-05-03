import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 9),  # 行動数: 3x3=9
        )

    def forward(self, x):
        # x: (batch, 3, 3, 3) -> (batch, 3, 3, 3)
        x = x.permute(0, 3, 1, 2)  # (batch, 3, 3, 3) -> (batch, 3, 3, 3)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # view の代わりに reshape を使用
        logits = self.fc(x)
        return logits

class REINFORCEAgent:
    def __init__(self, player_id, lr=1e-3, gamma=0.99):
        self.player_id = player_id
        self.gamma = gamma
        self.policy_net = PolicyNet()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def act(self, state, legal_actions):
        obs = self._state_to_obs(state)
        logits = self.policy_net(obs.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        mask = torch.zeros(9)
        mask[list(legal_actions)] = 1.0
        probs = probs * mask
        probs_sum = probs.sum()
        if probs_sum < 1e-8:
            # ほぼゼロの場合は一様分布にフォールバック
            probs = mask / mask.sum()
        else:
            probs = probs / probs_sum
        # 行動選択のみ detach して勾配を切る
        action = torch.multinomial(probs.detach(), 1).item()
        log_prob = torch.log(probs[action] + 1e-8)
        self.log_probs.append(log_prob)
        return action

    def update(self):
        # エピソード終了後に更新
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            loss += -log_prob * R
        loss = loss / len(self.log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def _state_to_obs(self, state):
        board = np.zeros((3, 3), dtype=int)
        history = state.history()
        for i, action in enumerate(history):
            player = i % 2
            row = action // 3
            col = action % 3
            board[row, col] = player + 1
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                if board[i, j] == 1:
                    obs[i, j, 0] = 1.0
                elif board[i, j] == 2:
                    obs[i, j, 1] = 1.0
                else:
                    obs[i, j, 2] = 1.0
        return torch.tensor(obs, dtype=torch.float32)

class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x_flat = x.reshape(x.size(0), -1)  # view の代わりに reshape を使用
        logits = self.actor_fc(x_flat)
        value = self.critic_fc(x_flat)
        return logits, value

class ActorCriticAgent:
    def __init__(self, player_id, lr=1e-3, gamma=0.99):
        self.player_id = player_id
        self.gamma = gamma
        self.net = ActorCriticNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []

    def act(self, state, legal_actions):
        obs = self._state_to_obs(state)
        with torch.no_grad():
            logits, value = self.net(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            mask = torch.zeros(9)
            mask[list(legal_actions)] = 1.0
            probs = probs * mask
            if probs.sum() == 0:
                probs = mask / mask.sum()
            else:
                probs = probs / probs.sum()
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action] + 1e-8)
            self.log_probs.append(log_prob)
            self.values.append(value.squeeze())
        return action

    def update(self):
        # エピソード終了後に更新
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = []
        for R, v in zip(returns, self.values):
            advantages.append(R - v.item())
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = 0
        critic_loss = 0
        for log_prob, adv, ret, v in zip(self.log_probs, advantages, returns, self.values):
            actor_loss += -log_prob * adv
            critic_loss += (ret - v) ** 2
        actor_loss = actor_loss / len(self.log_probs)
        critic_loss = critic_loss / len(self.values)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []

    def _state_to_obs(self, state):
        # REINFORCEAgent と同じ実装
        board = np.zeros((3, 3), dtype=int)
        history = state.history()
        for i, action in enumerate(history):
            player = i % 2
            row = action // 3
            col = action % 3
            board[row, col] = player + 1
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                if board[i, j] == 1:
                    obs[i, j, 0] = 1.0
                elif board[i, j] == 2:
                    obs[i, j, 1] = 1.0
                else:
                    obs[i, j, 2] = 1.0
        return torch.tensor(obs, dtype=torch.float32)