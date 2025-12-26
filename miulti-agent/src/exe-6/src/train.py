import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, List

# --- ハイパーパラメータ ---
GRID_SIZE = 5
OBS_SHAPE = 4    # (self_pos, self_hold, other_pos, other_hold)
STATE_SHAPE = 8  # 全エージェントの情報の統合
N_ACTIONS = 3    # 0: Stay, 1: Left, 2: Right
BATCH_SIZE = 32
GAMMA = 0.95
LR = 5e-4
MEMORY_CAPACITY = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_INTERVAL = 10
NUM_EPISODES = 500

# --- ネットワーク定義 ---

class MLPAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super(MLPAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim, hypernet_embed_dim):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        
        # ハイパーネットワーク: 状態からQMixerの重みを生成
        self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hypernet_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hypernet_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed_dim, mixing_embed_dim))
        
        self.hyper_b1 = nn.Linear(state_shape, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, mixing_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(mixing_embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.view(-1, self.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # 重みは常に正（単調性）を保証するためにabsを取る
        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, 32)
        b1 = self.hyper_b1(states).view(-1, 1, 32)
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(-1, 32, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        q_tot = torch.matmul(hidden, w2) + b2
        return q_tot.view(bs, -1)

# --- エージェントクラス ---

class QMixTrainer:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent_net = MLPAgent(OBS_SHAPE, 64, N_ACTIONS).to(self.device)
        self.mixer_net = QMixer(2, STATE_SHAPE, 32, 64).to(self.device)
        
        self.target_agent_net = MLPAgent(OBS_SHAPE, 64, N_ACTIONS).to(self.device)
        self.target_mixer_net = QMixer(2, STATE_SHAPE, 32, 64).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())
        
        self.optimizer = optim.Adam(list(self.agent_net.parameters()) + list(self.mixer_net.parameters()), lr=LR)
        self.memory = deque(maxlen=MEMORY_CAPACITY)

    def _get_tensors(self, obs, is_state=False):
        # 観測を正規化してテンソル化
        if is_state:
            state = []
            for i in range(2):
                state.extend([obs[i][0]/4.0, 1.0 if obs[i][1] else 0.0, obs[i][2]/4.0, 1.0 if obs[i][3] else 0.0])
            return torch.FloatTensor(state).to(self.device).unsqueeze(0)
        else:
            tensors = {}
            for i in range(2):
                o = obs[i]
                tensors[i] = torch.FloatTensor([o[0]/4.0, 1.0 if o[1] else 0.0, o[2]/4.0, 1.0 if o[3] else 0.0]).to(self.device).unsqueeze(0)
            return tensors

    def select_actions(self, obs, epsilon):
        if random.random() < epsilon:
            return {i: random.randint(0, N_ACTIONS-1) for i in range(2)}
        
        tensors = self._get_tensors(obs)
        actions = {}
        with torch.no_grad():
            for i in range(2):
                q_values = self.agent_net(tensors[i])
                actions[i] = q_values.argmax().item()
        return actions

    def train_step(self):
        if len(self.memory) < BATCH_SIZE: return 0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s_next, d = zip(*batch)
        
        states = torch.cat([self._get_tensors(obs, True) for obs in s])
        next_states = torch.cat([self._get_tensors(obs, True) for obs in s_next])
        rewards = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(d).to(self.device).unsqueeze(1)
        
        # 現在のQ値計算
        agent_qs = []
        for i in range(2):
            obs_i = torch.cat([self._get_tensors(obs)[i] for obs in s])
            q_vals = self.agent_net(obs_i)
            act_i = torch.LongTensor([action[i] for action in a]).to(self.device).unsqueeze(1)
            agent_qs.append(q_vals.gather(1, act_i))
        
        q_tot = self.mixer_net(torch.cat(agent_qs, dim=1), states)
        
        # ターゲットQ値計算
        with torch.no_grad():
            target_agent_qs = []
            for i in range(2):
                next_obs_i = torch.cat([self._get_tensors(obs)[i] for obs in s_next])
                target_q_vals = self.target_agent_net(next_obs_i)
                target_agent_qs.append(target_q_vals.max(1)[0].unsqueeze(1))
            
            target_q_tot = self.target_mixer_net(torch.cat(target_agent_qs, dim=1), next_states)
            y = rewards + GAMMA * target_q_tot * (1 - dones)
            
        loss = F.mse_loss(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# --- 学習メインループ ---

env = SwitchEnv(size=5)
trainer = QMixTrainer(env)
epsilon = EPS_START

print("🚀 学習開始...")
for ep in range(NUM_EPISODES):
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        actions = trainer.select_actions(obs, epsilon)
        next_obs, rewards, dones, _ = env.step(actions)
        
        # QMIXはチーム報酬の合計を学習対象にする
        team_reward = sum(rewards.values())
        trainer.memory.append((obs, actions, team_reward, next_obs, any(dones.values())))
        
        obs = next_obs
        total_reward += team_reward
        done = any(dones.values())
        trainer.train_step()
        
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    if ep % TARGET_UPDATE_INTERVAL == 0:
        trainer.target_agent_net.load_state_dict(trainer.agent_net.state_dict())
        trainer.target_mixer_net.load_state_dict(trainer.mixer_net.state_dict())
        
    if ep % 50 == 0:
        print(f"Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

print("✅ 学習完了。GIFを生成してください。")