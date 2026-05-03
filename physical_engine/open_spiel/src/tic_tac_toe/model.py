import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DiscretePolicyNetwork(nn.Module):
    """
    離散行動用の Actor（方策ネットワーク）
    出力は各行動の logits（カテゴリカル分布のパラメータ）
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def sample(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def log_prob(self, x, action):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        return log_prob

class QNetwork(nn.Module):
    """
    Critic（Qネットワーク）
    観測と行動（離散）を入力として Q(s, a) を出力
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # a は one-hot ベクトルとして与える想定
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
    
class DiscreteSACAgent:
    def __init__(
        self,
        obs_dim,        # 観測次元（例: 9）
        act_dim,        # 行動次元（例: 9）
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
        self.policy_net = DiscretePolicyNetwork(obs_dim, act_dim, hidden_dim)
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
                target_entropy = -np.log(1.0 / act_dim)  # 離散行動用の一般的な設定
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
                logits = self.policy_net(obs_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            else:
                action, _ = self.policy_net.sample(obs_tensor)
                action = action.item()
        return action

    def update(self, batch, batch_size):
        obs, act, rew, next_obs, done = batch

        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew).unsqueeze(-1)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done).unsqueeze(-1)

        # 行動を one-hot に変換
        act_onehot = F.one_hot(act, num_classes=self.act_dim).float()

        # Q関数の更新
        with torch.no_grad():
            next_act, next_log_prob = self.policy_net.sample(next_obs)
            next_act_onehot = F.one_hot(next_act, num_classes=self.act_dim).float()
            target_q1 = self.target_q_net1(next_obs, next_act_onehot)
            target_q2 = self.target_q_net2(next_obs, next_act_onehot)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target = rew + (1 - done) * self.gamma * target_q

        # Q1, Q2の損失
        q1 = self.q_net1(obs, act_onehot)
        q2 = self.q_net2(obs, act_onehot)
        q_loss1 = F.mse_loss(q1, target)
        q_loss2 = F.mse_loss(q2, target)
        q_loss = q_loss1 + q_loss2

        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q_optimizer1.step()
        self.q_optimizer2.step()

        # 方策の更新
        new_act, new_log_prob = self.policy_net.sample(obs)
        new_act_onehot = F.one_hot(new_act, num_classes=self.act_dim).float()
        q1_new = self.q_net1(obs, new_act_onehot)
        q2_new = self.q_net2(obs, new_act_onehot)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob.unsqueeze(-1) - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # αの自動調整（オプション）
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ターゲットネットワークのソフトアップデート
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item(), self.alpha

    def save_model(self, path):
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "q_net1_state_dict": self.q_net1.state_dict(),
            "q_net2_state_dict": self.q_net2.state_dict(),
            "target_q_net1_state_dict": self.target_q_net1.state_dict(),
            "target_q_net2_state_dict": self.target_q_net2.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer1_state_dict": self.q_optimizer1.state_dict(),
            "q_optimizer2_state_dict": self.q_optimizer2.state_dict(),
            "alpha": self.alpha,
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.q_net1.load_state_dict(checkpoint["q_net1_state_dict"])
        self.q_net2.load_state_dict(checkpoint["q_net2_state_dict"])
        self.target_q_net1.load_state_dict(checkpoint["target_q_net1_state_dict"])
        self.target_q_net2.load_state_dict(checkpoint["target_q_net2_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.q_optimizer1.load_state_dict(checkpoint["q_optimizer1_state_dict"])
        self.q_optimizer2.load_state_dict(checkpoint["q_optimizer2_state_dict"])
        self.alpha = checkpoint["alpha"]
        if self.auto_alpha and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        print(f"Model loaded from {path}")



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs),
            np.array(action),
            np.array(reward),
            np.array(next_obs),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)