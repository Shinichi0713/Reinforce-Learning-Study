import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class QMIXMemory:
    def __init__(self):
        self.obs = []          # 各エージェントの観測 (T, num_agents, obs_dim)
        self.actions = []      # 実行された行動 (T, num_agents)
        self.rewards = []      # 得られた報酬 (T, num_agents)
        self.next_obs = []     # 次の観測 (T, num_agents, obs_dim)
        self.dones = []        # 終了フラグ (T,)
        self.states = []       # グローバル状態 (T, state_dim)
        self.next_states = []  # 次のグローバル状態 (T, state_dim)

    def store(self, obs, actions, rewards, next_obs, done, state, next_state):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_obs.append(next_obs)
        self.dones.append(done)
        self.states.append(state)
        self.next_states.append(next_state)

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
        self.states = []
        self.next_states = []

    def get_batch(self):
        return {
            'obs': torch.stack(self.obs),
            'actions': torch.stack(self.actions),
            'rewards': torch.stack(self.rewards),
            'next_obs': torch.stack(self.next_obs),
            'dones': torch.tensor(self.dones, dtype=torch.float),
            'states': torch.stack(self.states),
            'next_states': torch.stack(self.next_states),
        }


class QNetwork(nn.Module):
    """各エージェント用のQネットワーク（パラメータ非共有）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class MixingNetwork(nn.Module):
    """QMIXのMixing Network（グローバルQ値を出力）"""
    def __init__(self, num_agents, state_dim, mixing_hidden=64):
        super().__init__()
        self.num_agents = num_agents
        # 状態依存の重み・バイアスを生成するネットワーク
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, num_agents * mixing_hidden)
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden)
        )
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, states):
        # agent_qs: (batch_size, num_agents)
        # states: (batch_size, state_dim)
        batch_size = agent_qs.size(0)
        
        # 第1層
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, -1)  # (b, num_agents, hidden)
        b1 = self.hyper_b1(states).unsqueeze(1)  # (b, 1, hidden)
        
        # 第2層
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, -1, 1)  # (b, hidden, 1)
        b2 = self.hyper_b2(states).unsqueeze(1)  # (b, 1, 1)
        
        # 混合計算（単調性を保証するためabsを使用）
        x = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (b, 1, hidden)
        q_total = torch.bmm(x, w2) + b2  # (b, 1, 1)
        return q_total.squeeze(-1).squeeze(-1)  # (b,)
    

class QMIXTrainer:
    def __init__(self, obs_dim, action_dim, num_agents=2, state_dim=None, gamma=0.95, lr=1e-3):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        if state_dim is None:
            state_dim = obs_dim * num_agents  # デフォルトは全観測の結合
        
        # 各エージェントのQネットワーク（パラメータ非共有）
        self.q_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim) for _ in range(num_agents)
        ])
        self.target_q_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim) for _ in range(num_agents)
        ])
        for target_q in self.target_q_nets:
            target_q.load_state_dict(self.q_nets[0].state_dict())  # 初期は同じパラメータ
        
        # Mixing Network
        self.mixing_net = MixingNetwork(num_agents, state_dim)
        self.target_mixing_net = MixingNetwork(num_agents, state_dim)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        
        # オプティマイザ
        q_params = list(self.q_nets.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(q_params, lr=lr)
        
        # ターゲットネットワークの更新用
        self.tau = 0.01  # ソフト更新係数
        
        # 探索用パラメータ（ボルツマン探索＋温度スケジュール）
        self.temperature = 1.0          # ボルツマン探索の温度（初期値）
        self.temperature_decay = 0.995  # 温度の減衰率（例：毎エピソード 0.995倍）
        self.min_temperature = 0.1      # 温度の下限

    def normalize_obs(self, obs_list):
        return torch.FloatTensor(np.array(obs_list))

    def train(self, memory):
        batch = memory.get_batch()
        obs = batch['obs']          # (T, num_agents, obs_dim)
        actions = batch['actions']   # (T, num_agents)
        rewards = batch['rewards']  # (T, num_agents)
        next_obs = batch['next_obs'] # (T, num_agents, obs_dim)
        dones = batch['dones']      # (T,)
        states = batch['states']     # (T, state_dim)
        next_states = batch['next_states'] # (T, state_dim)
        
        T = obs.size(0)
        
        # 現在のQ値の計算
        current_qs = []
        for i in range(self.num_agents):
            q_values = self.q_nets[i](obs[:, i])  # (T, action_dim)
            action_i = actions[:, i].long().unsqueeze(-1)  # (T, 1)
            q_i = q_values.gather(1, action_i).squeeze(-1)  # (T,)
            current_qs.append(q_i)
        current_qs = torch.stack(current_qs, dim=1)  # (T, num_agents)
        
        # ターゲットQ値の計算
        with torch.no_grad():
            next_qs = []
            for i in range(self.num_agents):
                next_q_values = self.target_q_nets[i](next_obs[:, i])  # (T, action_dim)
                next_q_i = next_q_values.max(1)[0]  # (T,)
                next_qs.append(next_q_i)
            next_qs = torch.stack(next_qs, dim=1)  # (T, num_agents)
            
            # グローバルターゲットQ値
            next_q_total = self.target_mixing_net(next_qs, next_states)  # (T,)
            
            # チーム報酬（平均）を使ってターゲットを計算
            team_rewards = rewards.mean(dim=1)  # (T,)
            target_q_total = team_rewards + self.gamma * (1 - dones) * next_q_total
        
        # 現在のグローバルQ値
        current_q_total = self.mixing_net(current_qs, states)  # (T,)
        
        # QMIX損失（MSE）
        loss_qmix = F.mse_loss(current_q_total, target_q_total.detach())
        
        # （オプション）エントロピー正則化を追加する場合
        # entropy_loss = 0
        # for i in range(self.num_agents):
        #     q_values = self.q_nets[i](obs[:, i])  # (T, action_dim)
        #     probs = F.softmax(q_values, dim=-1)
        #     log_probs = torch.log(probs + 1e-8)
        #     entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        #     entropy_loss -= entropy  # エントロピー最大化
        # total_loss = loss_qmix + 0.01 * entropy_loss
        
        total_loss = loss_qmix  # エントロピー正則化を使わない場合はこちら
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # ターゲットネットワークのソフト更新
        for target_q, q in zip(self.target_q_nets, self.q_nets):
            for tp, p in zip(target_q.parameters(), q.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        return total_loss.item()

    def select_action(self, obs_tensor, i, training=True):
        """
        QMIX用：ボルツマン探索で行動を選択
        training=True のときは温度を適用、False のときは argmax（評価用）
        """
        with torch.no_grad():
            q_values = self.q_nets[i](obs_tensor[i].unsqueeze(0))  # (1, action_dim)

            if training:
                # ボルツマン探索（softmax）
                probs = F.softmax(q_values / self.temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                # 評価時は argmax（活用）
                action = q_values.argmax().item()

        return action
    
    def update_temperature(self):
        """温度を減衰させる（エピソード終了時に呼ぶ）"""
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)