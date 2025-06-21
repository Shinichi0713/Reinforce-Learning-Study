import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- ネットワーク定義 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- main ---
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99
num_episodes = 1000
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = actor(state_tensor).squeeze(0).numpy()
        action = np.random.choice(action_dim, p=probs)
        next_state, reward, done, _, _ = env.step(action)

        # 記録
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        state = next_state
        episode_reward += reward

    # 学習（エピソード終了後に一括）
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    # Criticのターゲット（1-step TD）
    with torch.no_grad():
        next_state_values = critic(next_states_tensor)
        targets = rewards_tensor + gamma * next_state_values * (1 - dones_tensor)

    values = critic(states_tensor)
    critic_loss = nn.MSELoss()(values, targets)

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Advantage計算
    advantage = (targets - values).detach()

    # Actorの損失
    probs = actor(states_tensor)
    log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).clamp(min=1e-8))
    actor_loss = -(log_probs.squeeze() * advantage).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    reward_history.append(episode_reward)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, reward: {episode_reward}, avg(10): {np.mean(reward_history[-10:]):.1f}")

env.close()
# 学習曲線の表示
import matplotlib.pyplot as plt
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

