import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

# --- ハイパーパラメータ ---
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        # log_probの計算はSAC論文に準拠
        log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.max_action = max_action
        self.__load_networks()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval_mode:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob
            target_q = reward + (1 - done) * GAMMA * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        action_new, log_prob_new = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        actor_loss = (ALPHA * log_prob_new - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return actor_loss.item()

    def save_networks(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        torch.save(self.actor.state_dict(), f"{dir_current}/actor.pth")
        torch.save(self.critic.state_dict(), f"{dir_current}/critic.pth")
        torch.save(self.critic_target.state_dict(), f"{dir_current}/critic_target.pth")

    def __load_networks(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        try:
            self.actor.load_state_dict(torch.load(f"{dir_current}/actor.pth"))
            self.critic.load_state_dict(torch.load(f"{dir_current}/critic.pth"))
            self.critic_target.load_state_dict(torch.load(f"{dir_current}/critic_target.pth"))
            print("Networks loaded successfully.")
        except Exception:
            print("No pre-trained networks found. Starting from scratch.")