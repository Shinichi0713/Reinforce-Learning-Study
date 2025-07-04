import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random, os

# --- 環境クラス ---
class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("LunarLanderContinuous-v3")
        else:
            self.env = gym.make("LunarLanderContinuous-v3", render_mode="human")
        self.observation, self.info = self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.observation, self.info = self.env.reset()
        return self.observation

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

# --- SAC エージェント ---
class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, device):
        self.device = device
        self.act_limit = act_limit

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.alpha = 0.2
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
            action = action.cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            action = action.cpu().detach().numpy()[0]
        return action * self.act_limit

    def save_models(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.actor.cpu()
        self.critic1.cpu()
        self.critic2.cpu()
        torch.save(self.actor.state_dict(), f"{dir_current}/sac_actor.pth")
        torch.save(self.critic1.state_dict(), f"{dir_current}/sac_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{dir_current}/sac_critic2.pth")
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_action = next_action * self.act_limit
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(state)
        new_action = new_action * self.act_limit
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network update
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 追加: ロス値を返す
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

# --- 学習ループ ---
def train_sac():
    env = Environment(is_train=True)
    obs_dim = env.env.observation_space.shape[0]
    act_dim = env.env.action_space.shape[0]
    act_limit = float(env.env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(obs_dim, act_dim, act_limit, device)
    replay_buffer = ReplayBuffer(1000000)
    batch_size = 256
    total_steps = 200000
    start_steps = 10000
    update_after = 1000
    update_every = 50
    episode_rewards = []

    # 追加: ロス推移格納用リスト
    actor_losses = []
    critic1_losses = []
    critic2_losses = []

    state = env.reset()
    episode_reward = 0
    episode_steps = 0

    for t in range(total_steps):
        if t < start_steps:
            action = env.env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        episode_steps += 1

        if done:
            state = env.reset()
            episode_rewards.append(episode_reward)
            print(f"Step: {t}, Episode Reward: {episode_reward:.2f}")
            episode_reward = 0
            episode_steps = 0

        # 学習・ロス記録
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                actor_loss, critic1_loss, critic2_loss = agent.update(replay_buffer, batch_size)
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)

        agent.save_models()

    # --- 追加: ロス・報酬推移をファイル保存 ---
    dir_current = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(f"{dir_current}/actor_losses.txt", np.array(actor_losses))
    np.savetxt(f"{dir_current}/critic1_losses.txt", np.array(critic1_losses))
    np.savetxt(f"{dir_current}/critic2_losses.txt", np.array(critic2_losses))
    np.savetxt(f"{dir_current}/episode_rewards.txt", np.array(episode_rewards))

    env.close()

def eval_sac():
    env = Environment(is_train=False)
    obs_dim = env.env.observation_space.shape[0]
    act_dim = env.env.action_space.shape[0]
    act_limit = float(env.env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(obs_dim, act_dim, act_limit, device)
    dir_current = os.path.dirname(os.path.abspath(__file__))
    agent.actor.load_state_dict(torch.load(f"{dir_current}/sac_actor.pth"))
    agent.critic1.load_state_dict(torch.load(f"{dir_current}/sac_critic1.pth"))
    agent.critic2.load_state_dict(torch.load(f"{dir_current}/sac_critic2.pth"))

    for i in range(5):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        print(f"Episode Reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_sac()
    # eval_sac()

