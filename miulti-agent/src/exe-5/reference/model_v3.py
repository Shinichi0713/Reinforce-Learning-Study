import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class MAPPOMemory:
    def __init__(self):
        self.obs = []        # (T, NumAgents, ObsDim)
        self.states = []     # (T, ObsDim * NumAgents)
        self.actions = []    # (T, NumAgents)
        self.log_probs = []  # (T, NumAgents)
        self.rewards = []    # (T, NumAgents)
        self.dones = []      # (T)

    def store(self, obs, state, action, log_prob, reward, done):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def get_batch(self):
        return {
            'obs': torch.stack(self.obs),
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.stack(self.rewards),
            'dones': torch.tensor(self.dones, dtype=torch.float)
        }

# --- Actor (パラメータ共有) ---
class GRU_Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, h=None):
        # hがNoneならゼロ初期化
        if h is None:
            h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device)
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        probs = F.softmax(self.fc2(x), dim=-1)
        return Categorical(probs), h

# --- Critic (エージェントごとの価値を出力) ---
class GRU_Critic(nn.Module):
    def __init__(self, input_dim, num_agents, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, num_agents)  # 各エージェントの価値を出力

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device)
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        values = self.fc2(x)  # shape: (batch, seq_len, num_agents)
        return values, h

# --- MAPPO Trainer (修正版) ---
class MAPPOTrainer:
    def __init__(self, obs_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.gamma = 0.95
        self.clip_eps = 0.2
        
        # Actor: 入力 = 観測 + agent_id (one-hot)
        self.actor = GRU_Actor(obs_dim + num_agents, action_dim)
        # Critic: 入力 = 全エージェントの観測を結合, 出力 = 各エージェントの価値
        self.critic = GRU_Critic(obs_dim * num_agents, num_agents)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=2e-3)

    def normalize_obs(self, obs_list):
        return torch.FloatTensor(np.array(obs_list))

    def train(self, memory):
        obs = torch.stack(memory.obs)        # (T, num_agents, obs_dim)
        states = torch.stack(memory.states)  # (T, obs_dim * num_agents)
        actions = torch.stack(memory.actions) # (T, num_agents)
        old_log_probs = torch.stack(memory.log_probs) # (T, num_agents)
        rewards = torch.stack(memory.rewards) # (T, num_agents)
        dones = torch.tensor(memory.dones, dtype=torch.float) # (T,)
        
        T = obs.size(0)
        
        # --- Critic の更新 ---
        # Critic入力: (batch=1, seq_len=T, input_dim=obs_dim*num_agents)
        states_batch = states.unsqueeze(0)
        values, _ = self.critic(states_batch)  # values: (1, T, num_agents)
        values = values.squeeze(0)             # (T, num_agents)
        
        # 累積報酬の計算 (エージェントごと)
        returns = torch.zeros_like(rewards)
        running_returns = torch.zeros(self.num_agents)
        for t in reversed(range(T)):
            running_returns = rewards[t] + self.gamma * running_returns
            returns[t] = running_returns
        
        # Advantage = returns - values
        advantages = returns - values.detach()  # (T, num_agents)
        
        # Critic loss: 各エージェントの価値と累積報酬のMSE
        critic_loss = F.mse_loss(values, returns.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor の更新 (パラメータ共有) ---
        actor_loss_total = 0
        
        # 各エージェントの観測にagent_idを結合
        agent_ids = torch.eye(self.num_agents, device=obs.device)  # (num_agents, num_agents)
        agent_ids_expanded = agent_ids.unsqueeze(0).expand(T, -1, -1)  # (T, num_agents, num_agents)
        
        # 観測とagent_idを結合
        obs_with_id = torch.cat([obs, agent_ids_expanded], dim=-1)  # (T, num_agents, obs_dim+num_agents)
        
        # Actor入力: (batch=1, seq_len=T, input_dim=obs_dim+num_agents)
        obs_batch = obs_with_id.transpose(0, 1).unsqueeze(0)  # (1, num_agents, T, input_dim)
        # RNNの入力形状に合わせるため、num_agentsごとに独立に処理するか、
        # あるいは (1, T*num_agents, input_dim) にreshapeして一括処理する
        # ここでは簡略化のため、エージェントごとにループ
        for i in range(self.num_agents):
            agent_obs = obs_with_id[:, i].unsqueeze(0)  # (1, T, input_dim)
            dist, _ = self.actor(agent_obs)
            new_log_probs = dist.log_prob(actions[:, i])  # (T,)
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            surr1 = ratio * advantages[:, i]
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[:, i]
            
            actor_loss_total += -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

        self.actor_opt.zero_grad()
        actor_loss_total.backward()
        self.actor_opt.step()