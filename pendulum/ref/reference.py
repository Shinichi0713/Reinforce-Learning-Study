import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Pendulumのアクションを離散化
DISCRETE_ACTIONS = np.array([[-2.0], [0.0], [2.0]])  # 左, ニュートラル, 右

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

def select_action(q_net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(DISCRETE_ACTIONS))
    else:
        state_v = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_net(state_v)
        return torch.argmax(q_values).item()

# --- メイン ---
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = len(DISCRETE_ACTIONS)
q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(10000)

num_episodes = 200
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
update_target_every = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(200):
        # env.render()  # 必要なら描画
        action_idx = select_action(q_net, state, epsilon)
        action = DISCRETE_ACTIONS[action_idx]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action_idx, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # 学習
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states_v = torch.FloatTensor(states)
            actions_v = torch.LongTensor(actions)
            rewards_v = torch.FloatTensor(rewards)
            next_states_v = torch.FloatTensor(next_states)
            dones_v = torch.FloatTensor(dones)

            q_values = q_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states_v).max(1)[0]
                targets = rewards_v + gamma * next_q_values * (1 - dones_v)
            loss = nn.MSELoss()(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # ターゲットネットワークの更新
    if episode % update_target_every == 0:
        target_net.load_state_dict(q_net.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

env.close()
