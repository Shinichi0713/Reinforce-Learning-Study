import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class JobShopEnv(gym.Env):
    def __init__(self, n_jobs=5, n_machines=2):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.action_space = spaces.Discrete(n_jobs * n_machines)  # (job, machine)の組み合わせ
        # 状態: [各ジョブの所要時間(5), 割当済みフラグ(5), 各マシンの累積時間(2)]
        self.observation_space = spaces.Box(low=0, high=20, shape=(n_jobs*2 + n_machines,), dtype=np.float32)

    def reset(self):
        self.process_times = np.random.randint(1, 10, size=(self.n_jobs,))
        self.assigned = np.zeros(self.n_jobs, dtype=bool)
        self.machine_times = np.zeros(self.n_machines, dtype=np.float32)
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.process_times, self.assigned.astype(np.float32), self.machine_times])

    def step(self, action):
        job = action // self.n_machines
        machine = action % self.n_machines

        reward = 0
        if self.assigned[job]:
            # すでに割り当て済みジョブを選んだらペナルティ
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, {}

        # 割り当て
        self.assigned[job] = True
        self.machine_times[machine] += self.process_times[job]

        if self.assigned.all():
            # 全ジョブ割り当て完了
            makespan = self.machine_times.max()
            reward = -makespan  # makespanが小さいほど報酬大
            self.done = True
        else:
            reward = 0
            self.done = False

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        print(f"Times:{self.process_times}, Assigned:{self.assigned}, MachineTimes:{self.machine_times}")



class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.qnet = QNetwork(obs_dim, action_dim)
        self.target_qnet = QNetwork(obs_dim, action_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = 64
        self.update_steps = 0

    def act(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.qnet.fc[-1].out_features)
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.qnet(obs)
        return q_vals.argmax().item()

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        q = self.qnet(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_qnet(s_).max(1, keepdim=True)[0]
            q_target = r + self.gamma * q_next * (1 - d)
        loss = nn.MSELoss()(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 100 == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())


def train():
    env = JobShopEnv()
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    num_episodes = 5000

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(obs, epsilon=max(0.05, 0.5 - ep/4000))
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            total_reward += reward
        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1} reward: {total_reward}")


def eval():
    env = JobShopEnv()
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.act(obs, epsilon=0.0)
        obs, reward, done, _ = env.step(action)
        env.render()
    print(f"最終makespan: {-reward}")


if __name__ == "__main__":
    train()

