import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前処理
def preprocess(obs):
    obs = obs.astype(np.float32) / 255.0  # 0-1正規化
    return obs

# ポリシーネットワーク
class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.fc = nn.Linear(64 * 20 * 20, 128)
        self.mean_head = nn.Linear(128, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.log_std)
        return mean, std

# バリューネットワーク
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.fc = nn.Linear(64 * 20 * 20, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        value = self.value_head(x)
        return value

env = gym.make('CarRacing-v2', continuous=True, render_mode=None)
obs_shape = (3, 96, 96)
num_actions = env.action_space.shape[0]
action_low = torch.tensor(env.action_space.low, device=device)
action_high = torch.tensor(env.action_space.high, device=device)

policy_net = PolicyNet(num_actions).to(device)
value_net = ValueNet().to(device)
optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

gamma = 0.99
epsilon = 0.2
value_scale = 0.5
entropy_scale = 0.01

def select_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).permute(0,3,1,2)
    mean, std = policy_net(obs)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    action_clipped = torch.clamp(action, action_low, action_high)
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action_clipped.cpu().numpy()[0], log_prob.item()

for episode in range(1000):
    obs = preprocess(env.reset()[0])
    done = False
    states, actions, rewards, values, log_probs = [], [], [], [], []
    score = 0

    while not done:
        value = value_net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).permute(0,3,1,2)).item()
        action, log_prob = select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = preprocess(next_obs)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        obs = next_obs
        score += reward

        if done or len(states) >= 2048:
            next_value = value_net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).permute(0,3,1,2)).item()
            returns = []
            R = next_value
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = np.array(returns)
            values_np = np.array(values)
            advantages = returns - values_np

            # PPO学習
            states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device).permute(0,3,1,2)
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
            old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            mean, std = policy_net(states_t)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            ratio = (new_log_probs - old_log_probs_t).exp()
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_pred = value_net(states_t).squeeze(1)
            value_loss = F.mse_loss(value_pred, returns_t) * value_scale
            entropy_loss = -entropy.mean() * entropy_scale
            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Episode {episode}, Score: {score:.2f}, Loss: {loss.item():.4f}")
            break  # 1エピソードごとに学習


