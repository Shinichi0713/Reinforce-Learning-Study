import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random, os

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# --- CNN特徴抽出 ---
class CNNFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # 96x96x3 -> (出力特徴量数)を計算
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            n_flat = self.cnn(dummy).shape[1]
        self.n_flat = n_flat
    def forward(self, x):
        x = x / 255.0  # 画像正規化
        return self.cnn(x)

# --- Actor ---
class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.feature = CNNFeature()
        self.fc = nn.Sequential(
            nn.Linear(self.feature.n_flat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, action_dim)
        self.logstd_head = nn.Linear(128, action_dim)
    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        mu = self.mu_head(x)
        logstd = self.logstd_head(x).clamp(-4, 1)
        return mu, logstd
    def sample(self, x):
        mu, logstd = self.forward(x)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- Critic (Qネットワーク) ---
class Critic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.feature = CNNFeature()
        self.fc = nn.Sequential(
            nn.Linear(self.feature.n_flat + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)
        return self.fc(x)

# --- SACエージェント ---
class SACAgent:
    def __init__(self, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.actor = Actor(action_dim).to(device)
        self.critic1 = Critic(action_dim).to(device)
        self.critic2 = Critic(action_dim).to(device)
        self.target_critic1 = Critic(action_dim).to(device)
        self.target_critic2 = Critic(action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), 1e-4)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), 1e-3)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), 1e-3)
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # 固定値
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device)
        if eval:
            with torch.no_grad():
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]
    def update(self, buffer, batch_size):
        state, action, reward, next_state, done = buffer.sample(batch_size)
        state = torch.FloatTensor(state).permute(0,3,1,2).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0,3,1,2).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Critic更新
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = reward + (1 - done) * self.gamma * target_q
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(q1, target)
        critic2_loss = nn.MSELoss()(q2, target)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Actor更新
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ターゲットネットワーク更新
        for target, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)
        for target, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)

# --- 学習ループ ---
def train_sac_car_racing(
    num_episodes=5,
    max_steps=1000,
    batch_size=16,
    start_steps=1000,
    update_after=1000,
    update_every=50
):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = env.action_space.shape[0]
    agent = SACAgent(action_dim, device)
    buffer = ReplayBuffer()
    total_steps = 0

    # 保存先ディレクトリ
    save_dir = "sac_params"
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        for step in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            total_steps += 1
            if total_steps > update_after and len(buffer) > batch_size and total_steps % update_every == 0:
                for _ in range(update_every):
                    agent.update(buffer, batch_size)
            if done:
                break
        print(f"Episode {episode+1}: Return={episode_return:.1f}")

        # --- ここでパラメータを保存 ---
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"actor_ep{episode+1}.pth"))
        torch.save(agent.critic1.state_dict(), os.path.join(save_dir, f"critic1_ep{episode+1}.pth"))
        torch.save(agent.critic2.state_dict(), os.path.join(save_dir, f"critic2_ep{episode+1}.pth"))
        torch.save(agent.target_critic1.state_dict(), os.path.join(save_dir, f"target_critic1_ep{episode+1}.pth"))
        torch.save(agent.target_critic2.state_dict(), os.path.join(save_dir, f"target_critic2_ep{episode+1}.pth"))

    env.close()

if __name__ == "__main__":
    train_sac_car_racing(num_episodes=1000)
