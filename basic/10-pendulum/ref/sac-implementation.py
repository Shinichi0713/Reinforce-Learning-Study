import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, os

# --- パラメータ ---
ENV_NAME = "Pendulum-v1"
SEED = 42
EPISODES = 300
MAX_STEPS = 300
BATCH_SIZE = 256
MEMORY_SIZE = 1000000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2  # エントロピー正則化係数

# --- ユーティリティ ---
def set_seed(env, seed=42):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

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

# --- ネットワーク ---
LOG_STD_MIN = -20
LOG_STD_MAX = 2

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

# --- SACエージェント ---
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

    def save_networks(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        torch.save(self.actor.state_dict(), f"{dir_current}/actor.pth")
        torch.save(self.critic.state_dict(), f"{dir_current}/critic.pth")
        torch.save(self.critic_target.state_dict(), f"{dir_current}/critic_target.pth")

    def __load_networks(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.actor.load_state_dict(torch.load(f"{dir_current}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{dir_current}/critic.pth"))
        self.critic_target.load_state_dict(torch.load(f"{dir_current}/critic_target.pth"))
        print("Networks loaded successfully.")

# --- メインループ ---
def train():
    env = gym.make(ENV_NAME)
    set_seed(env, SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SACAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    returns = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_return = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward

            if len(replay_buffer) > BATCH_SIZE:
                agent.update(replay_buffer, BATCH_SIZE)

            if done:
                break

        returns.append(episode_return)
        if episode % 10 == 0:
            avg_return = np.mean(returns[-10:])
            print(f"Episode {episode}: Return {episode_return:.2f}, Avg(10) {avg_return:.2f}")
    agent.save_networks()
    print("Training completed.")
    env.close()


def eval():
    env = gym.make(ENV_NAME, render_mode="human")
    set_seed(env, SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SACAgent(state_dim, action_dim, max_action)

    state, _ = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Total Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    # train()
    eval()
