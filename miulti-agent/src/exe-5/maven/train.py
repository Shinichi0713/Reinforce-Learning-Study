import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Tuple, Dict, Optional

# --- MAVEN風のニューラルネットワーク ---
class QNetwork(nn.Module):
    """各エージェント用のQネットワーク（観測 → Q値）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)


class MixingNetwork(nn.Module):
    """
    MAVEN風のMixing Network（簡易版）
    - 入力: 各エージェントのQ値 + latent variable z
    - 出力: 共同Q値 Q_tot
    """
    def __init__(self, n_agents, hidden_dim=64, z_dim=4):
        super().__init__()
        self.n_agents = n_agents
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(n_agents + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, z):
        # agent_qs: (batch, n_agents)
        # z: (batch, z_dim)
        x = torch.cat([agent_qs, z], dim=-1)
        return self.net(x)  # (batch, 1)


# --- 経験再生バッファ ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_list, actions, rewards, next_obs_list, done):
        self.buffer.append((obs_list, actions, rewards, next_obs_list, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class MavenTrainer:
    def __init__(self, n_agents, obs_dim, action_dim, lr=1e-3, gamma=0.99,
                 hidden_dim=64, z_dim=4, target_update_interval=500, tau=0.01):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.z_dim = z_dim
        self.target_update_interval = target_update_interval
        self.tau = tau  # ソフトアップデート用
        self.update_count = 0

        # 各エージェントのQネットワーク
        self.q_nets = [QNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        self.target_q_nets = [QNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        for i in range(n_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())

        # Mixing Network
        self.mixing_net = MixingNetwork(n_agents, hidden_dim, z_dim)
        self.target_mixing_net = MixingNetwork(n_agents, hidden_dim, z_dim)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # オプティマイザ
        all_params = list(self.mixing_net.parameters())
        for q_net in self.q_nets:
            all_params.extend(list(q_net.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr)

    def sample_z(self, batch_size):
        # 一様分布に変更（探索の多様性を高める）
        return torch.rand(batch_size, self.z_dim) * 2 - 1  # [-1, 1]

    def compute_q_tot(self, obs_batch, action_batch, z, nets, mixing_net):
        batch_size = len(obs_batch)
        agent_qs = []
        for i in range(self.n_agents):
            obs_i = torch.FloatTensor([obs[i] for obs in obs_batch])
            actions_i = torch.LongTensor([a[i] for a in action_batch])
            q_values = nets[i](obs_i)
            q_i = q_values.gather(1, actions_i.unsqueeze(1)).squeeze(1)
            agent_qs.append(q_i)
        agent_qs = torch.stack(agent_qs, dim=1)
        q_tot = mixing_net(agent_qs, z)
        return q_tot.squeeze(1)

    def update(self, batch_size, buffer: ReplayBuffer):
        if len(buffer) < batch_size:
            return

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.sample(batch_size)
        batch_size = len(obs_batch)

        # 各 update ごとに新しい z をサンプリング（QMIX風に近づける）
        z = self.sample_z(batch_size)

        # 現在のQ_tot
        current_q_tot = self.compute_q_tot(obs_batch, action_batch, z, self.q_nets, self.mixing_net)

        # ターゲットQ_tot（同じ z を使う）
        with torch.no_grad():
            next_agent_qs = []
            for i in range(self.n_agents):
                next_obs_i = torch.FloatTensor([next_obs[i] for next_obs in next_obs_batch])
                next_q_values = self.target_q_nets[i](next_obs_i)
                next_max_q = next_q_values.max(1)[0]
                next_agent_qs.append(next_max_q)
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            next_q_tot = self.target_mixing_net(next_agent_qs, z)
            rewards = torch.FloatTensor([sum(r) for r in reward_batch])
            dones = torch.FloatTensor(done_batch)
            target_q_tot = rewards + (1 - dones) * self.gamma * next_q_tot.squeeze(1)

        # TD誤差（Huber loss）
        loss = nn.SmoothL1Loss()(current_q_tot, target_q_tot)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ソフトアップデート（毎回少しずつターゲットを更新）
        self.update_count += 1
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_q_nets[i].parameters(), self.q_nets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def select_actions(self, obs_list, epsilon=0.1):
        actions = []
        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                action = np.random.randint(self.action_dim)
            else:
                obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_nets[i](obs_tensor)
                    action = q_values.argmax().item()
            actions.append(action)
        return actions

def train_maven(env, trainer, buffer, episodes=500, batch_size=32,
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.998):
    epsilon = epsilon_start
    rewards_history = []

    for ep in range(episodes):
        positions, obs_list = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < env.max_steps:
            actions = trainer.select_actions(obs_list, epsilon=epsilon)
            next_positions, next_obs_list, rewards, done, info = env.step(actions)
            buffer.push(obs_list, actions, rewards, next_obs_list, done)
            loss = trainer.update(batch_size, buffer)  # z は update 内でサンプリング
            obs_list = next_obs_list
            total_reward += sum(rewards)
            step_count += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return rewards_history


# --- GIF保存（前回の save_gif を trainer 対応に拡張） ---
def save_gif(env, trainer, filename="cooperative_grid_maven.gif", max_steps=20):
    frames = []
    positions, obs_list = env.reset()
    done = False
    step_count = 0

    fig, ax = plt.subplots(figsize=(5, 5))

    while not done and step_count < max_steps:
        env.render(ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))

        actions = trainer.select_actions(obs_list, epsilon=0.0)  # 評価時はε=0
        positions, obs_list, rewards, done, info = env.step(actions)
        step_count += 1

    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0
        )
        plt.close(fig)
        print(f"✅ GIF saved as {filename}")
    else:
        plt.close(fig)
        print("❌ No frames to save")
