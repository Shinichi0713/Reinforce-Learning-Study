import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MAPPOMemory:
    def __init__(self):
        # データの保存用リスト
        self.obs = []        # 各エージェントの個別観測 (T, NumAgents, ObsDim)
        self.states = []     # 集中Critic用のグローバル状態 (T, ObsDim * NumAgents)
        self.actions = []    # 実行された行動 (T, NumAgents)
        self.log_probs = []  # 行動の対数確率 (T, NumAgents)
        self.rewards = []    # 得られた報酬 (T, NumAgents)
        self.dones = []      # 終了フラグ (T)
        
        # RNN (GRU) の初期隠れ状態を保存（学習の開始点として必要）
        self.h_actors = []   # 各エージェントのActor初期隠れ状態
        self.h_critics = []  # Criticの初期隠れ状態

    def store(self, obs, state, action, log_prob, reward, done):
        """1ステップ分のデータを保存"""
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """学習後にメモリを空にする"""
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.h_actors = []
        self.h_critics = []

    def get_batch(self):
        """保存されたリストをテンソルに変換して返す（デバッグや拡張用）"""
        return {
            'obs': torch.stack(self.obs),
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.stack(self.rewards),
            'dones': torch.tensor(self.dones, dtype=torch.float)
        }

  
# --- 1. ネットワーク定義 (パラメータ共有用) ---
class GRU_Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        probs = F.softmax(self.fc2(x), dim=-1)
        return Categorical(probs), h

class GRU_Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        value = self.fc2(x)
        return value, h

# --- 2. MAPPO Trainer ---
class MAPPOTrainer:
    def __init__(self, obs_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.gamma = 0.95
        self.clip_eps = 0.2
        
        # Actorはパラメータ共有 (入力: obs + agent_id)
        self.actor = GRU_Actor(obs_dim + num_agents, action_dim)
        # Criticは集中型 (入力: 全員のobsを結合)
        self.critic = GRU_Critic(obs_dim * num_agents)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=2e-3)

    def normalize_obs(self, obs_list):
        return torch.FloatTensor(np.array(obs_list))

    def train(self, memory):
        obs = torch.stack(memory.obs) # (T, NumAgents, ObsDim)
        states = torch.stack(memory.states).unsqueeze(0) # (1, T, ObsDim*NumAgents)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        rewards = torch.stack(memory.rewards)
        
        T = obs.size(0)
        
        # 累積報酬の計算
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(self.num_agents)
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # 集中Criticの更新
        values, _ = self.critic(states, memory.h_critics[0])
        values = values.squeeze()
        target_returns = returns.mean(dim=-1) # チームの平均報酬
        
        critic_loss = F.mse_loss(values, target_returns.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # パラメータ共有Actorの更新
        advantages = (target_returns - values.detach()).unsqueeze(-1).repeat(1, self.num_agents)
        
        actor_loss_total = 0
        for i in range(self.num_agents):
            agent_id = torch.zeros(T, self.num_agents)
            agent_id[:, i] = 1.0
            combined_obs = torch.cat([obs[:, i], agent_id], dim=-1).unsqueeze(0)
            
            dist, _ = self.actor(combined_obs, memory.h_actors[0][i])
            new_log_probs = dist.log_prob(actions[:, i])
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            surr1 = ratio * advantages[:, i]
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[:, i]
            
            actor_loss_total += -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

        self.actor_opt.zero_grad()
        actor_loss_total.backward()
        self.actor_opt.step()
