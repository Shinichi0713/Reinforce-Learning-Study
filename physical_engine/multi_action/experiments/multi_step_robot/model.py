import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, act_dim)
        self.fc_log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # [-1,1] に収める
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        action = mu + noise * std
        log_prob = (-0.5 * (noise ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(dim=-1)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, obs_dim=19, act_dim=2, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        return obs, act, rew, next_obs, done

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        obs_dim,        # 観測次元（例: 19）
        act_dim,        # 行動次元（例: 2）
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        target_entropy=None,
        alpha_lr=3e-4,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # ネットワーク
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_dim)
        self.q_net1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q_net2 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_q_net1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_q_net2 = QNetwork(obs_dim, act_dim, hidden_dim)

        # ターゲットネットワークを初期化
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # オプティマイザ
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=lr)

        # エントロピー係数α
        self.alpha = alpha
        if auto_alpha:
            if target_entropy is None:
                target_entropy = -act_dim  # 一般的な設定
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

    def get_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.policy_net(obs_tensor)
                action = mu.squeeze(0).numpy()
            else:
                action, _ = self.policy_net.sample(obs_tensor)
                action = action.squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)

    def update(self, batch, batch_size):
        obs, act, rew, next_obs, done = batch

        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew).unsqueeze(-1)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done).unsqueeze(-1)

        # Q関数の更新
        with torch.no_grad():
            next_act, next_log_prob = self.policy_net.sample(next_obs)
            target_q1 = self.target_q_net1(next_obs, next_act)
            target_q2 = self.target_q_net2(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target = rew + (1 - done) * self.gamma * target_q

        # Q1, Q2の損失
        q1 = self.q_net1(obs, act)
        q2 = self.q_net2(obs, act)
        q_loss1 = F.mse_loss(q1, target)
        q_loss2 = F.mse_loss(q2, target)
        q_loss = q_loss1 + q_loss2

        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_loss.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.step()

        # 方策の更新
        new_act, new_log_prob = self.policy_net.sample(obs)
        q1_new = self.q_net1(obs, new_act)
        q2_new = self.q_net2(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob.unsqueeze(-1) - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # αの自動調整（オプション）
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ターゲットネットワークのソフトアップデート
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item(), self.alpha
