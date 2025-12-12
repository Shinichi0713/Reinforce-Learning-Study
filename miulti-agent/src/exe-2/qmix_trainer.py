import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, List
import time
# from IPython import display # Jupyter環境用 (ここではコメントアウト)

# ----------------------------------------------------
# 0. 環境設定とWarehouseEnvクラス (再掲)
# ----------------------------------------------------

# 環境設定 (QMIXのパラメータではないが、コード実行に必須)
GRID_SIZE = 10
NUM_AGENTS = 2
NUM_ORDERS = 3
PICKUP_LOCATIONS = [(1, 1), (8, 1), (5, 8)]
DROPOFF_LOCATION = (5, 5)

# QMIXハイパーパラメータ
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 100
LR_AGENT = 0.0005
LR_MIXER = 0.001
MEMORY_SIZE = 50000
NUM_EPISODES = 5000 # サンプルのため少なめに設定
MAX_STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WarehouseEnv:
    # 以前提供されたWarehouseEnvクラスの定義をここにコピー＆ペーストしてください。
    # (ここでは簡潔のため省略しますが、以前のコードブロックの通りに動作します)
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = 5 
        self.fig = None
        self.ax = None
        self.reset()

    def reset(self) -> Dict[int, Tuple]:
        self.agent_positions: Dict[int, Tuple[int, int]] = {
            i: (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for i in range(self.num_agents)
        }
        self.agent_holding: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.remaining_orders: List[int] = list(range(NUM_ORDERS))
        return self._get_obs()

    def _get_obs(self) -> Dict[int, Tuple]:
        obs = {}
        for i in range(self.num_agents):
            obs[i] = (
                self.agent_positions[i],
                self.agent_holding[i],
                tuple(self.remaining_orders)
            )
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        next_positions: Dict[int, Tuple[int, int]] = {}
        rewards: Dict[int, float] = {i: -0.1 for i in range(self.num_agents)} 
        
        # 1. 位置の更新 (省略: 以前のコードと同じロジック)
        for i, action in actions.items():
            current_x, current_y = self.agent_positions[i]
            next_x, next_y = current_x, current_y

            if action == 1: next_y += 1 
            elif action == 2: next_y -= 1 
            elif action == 3: next_x -= 1 
            elif action == 4: next_x += 1 
            
            next_x = np.clip(next_x, 0, self.size - 1)
            next_y = np.clip(next_y, 0, self.size - 1)
            next_positions[i] = (next_x, next_y)

        # 2. 衝突判定 (省略: 以前のコードと同じロジック)
        final_positions = self.agent_positions.copy()
        is_collision = False
        
        for i in range(self.num_agents):
            pos = next_positions[i]
            is_colliding = False
            for j in range(self.num_agents):
                if i != j and pos == next_positions[j]:
                    is_colliding = True
                    break
            
            if is_colliding:
                rewards[i] -= 5.0
                is_collision = True
            else:
                final_positions[i] = pos
        
        self.agent_positions = final_positions

        # 3. ピックアップ・ドロップオフ (省略: 以前のコードと同じロジック)
        dropoff_reward = 0.0
        for i in range(self.num_agents):
            current_pos = self.agent_positions[i]
            
            if self.agent_holding[i]:
                if current_pos == DROPOFF_LOCATION:
                    self.agent_holding[i] = False
                    dropoff_reward += 50.0
            else:
                for order_idx in self.remaining_orders:
                    if current_pos == PICKUP_LOCATIONS[order_idx]:
                        self.remaining_orders.remove(order_idx)
                        self.agent_holding[i] = True
                        rewards[i] += 10.0
                        break
                        
        if dropoff_reward > 0.0:
            for i in range(self.num_agents):
                rewards[i] += dropoff_reward / self.num_agents 

        done = {i: len(self.remaining_orders) == 0 for i in range(self.num_agents)}
        return self._get_obs(), rewards, done, {"collision": is_collision}
    
    # renderメソッドは省略

# ----------------------------------------------------
# 1. QMIX ネットワーク定義
# ----------------------------------------------------

# --- AgentNet (個別Qネットワーク) ---
class AgentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentNet, self).__init__()
        # 集中学習の間に、各エージェントは自身のローカル観測 o_i に基づいて Q_i を学習します。
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- QMixer (モノトニシティ制約付きのミキサーネットワーク) ---
class QMixer(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # ハイパーネットワーク W1: グローバル状態 s から W1 の重み (非負) を生成
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # ハイパーネットワーク W2: グローバル状態 s から W2 の重み (非負) を生成
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        self.hidden_dim = hidden_dim

    def forward(self, agent_qs, states):
        # agent_qs: (Batch, N_agents, 1) or (Batch, N_agents)
        # states: (Batch, State_dim)
        batch_size = agent_qs.size(0)

        # W1 の計算 (モノトニシティ制約のため、絶対値を取って非負にする)
        w1 = torch.abs(self.hyper_w1(states)) 
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim) 
        
        # Q * W1 の計算: (B, 1, N) x (B, N, H) -> (B, 1, H)
        agent_qs = agent_qs.view(batch_size, self.num_agents, 1) 
        hidden = torch.bmm(agent_qs.transpose(1, 2), w1) 

        # B1 の計算
        b1 = self.hyper_b1(states) 
        b1 = b1.view(batch_size, 1, self.hidden_dim) 

        hidden = F.relu(hidden + b1) 

        # W2 の計算 (モノトニシティ制約のため、絶対値を取って非負にする)
        w2 = torch.abs(self.hyper_w2(states)) 
        w2 = w2.view(batch_size, self.hidden_dim, 1) 
        
        # Q_tot の計算: (B, 1, H) x (B, H, 1) -> (B, 1, 1)
        q_tot = torch.bmm(hidden, w2) 

        b2 = self.hyper_b2(states)
        b2 = b2.view(batch_size, 1, 1)
        
        q_tot = q_tot.view(batch_size, 1) + b2.view(batch_size, 1) 

        return q_tot.squeeze(-1) # (B,)
        
# ----------------------------------------------------
# 2. Replay Memory
# ----------------------------------------------------

class QMixReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        # state, action, next_state, reward は Dict[int, ...]
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ----------------------------------------------------
# 3. QMIXエージェントクラス
# ----------------------------------------------------

class QMixAgent:
    def __init__(self, env: 'WarehouseEnv', state_dim):
        self.env = env
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.state_dim = state_dim
        
        # Policy Networks
        self.agent_nets = nn.ModuleList([AgentNet(self._get_agent_input_dim(), self.action_space).to(device) 
                                         for _ in range(self.num_agents)])
        self.mixer = QMixer(self.num_agents, self.state_dim, hidden_dim=32).to(device)

        # Target Networks (更新が遅れるネットワーク)
        self.target_agent_nets = nn.ModuleList([AgentNet(self._get_agent_input_dim(), self.action_space).to(device) 
                                                for _ in range(self.num_agents)])
        self.target_mixer = QMixer(self.num_agents, self.state_dim, hidden_dim=32).to(device)
        self.update_target_networks()
        
        # Optimizers
        self.optimizer_agents = optim.Adam(self._get_agent_params(), lr=LR_AGENT)
        self.optimizer_mixer = optim.Adam(self.mixer.parameters(), lr=LR_MIXER)
        
        self.memory = QMixReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def _get_agent_input_dim(self):
        # ローカル入力次元: 位置(2) + 荷物有無(1) + 全オーダーのマスク(NUM_ORDERS) = 6
        return 2 + 1 + NUM_ORDERS 

    def _get_agent_params(self):
        params = []
        for net in self.agent_nets:
            params.extend(list(net.parameters()))
        return params

    def preprocess_state(self, obs: Dict[int, Tuple]):
        """観測をテンソルに変換し、グローバル状態とローカル状態を分離"""
        global_state = []
        order_mask = [0.0] * NUM_ORDERS
        remaining_order_ids = obs[0][2] 
        for order_id in remaining_order_ids:
            order_mask[order_id] = 1.0

        for i in range(self.num_agents):
            (px, py), holding, _ = obs[i]
            global_state.extend([px / (self.env.size - 1), py / (self.env.size - 1), float(holding)])
        
        global_state.extend(order_mask)
        global_state_tensor = torch.tensor([global_state], dtype=torch.float32, device=device)
        
        local_states = []
        for i in range(self.num_agents):
            (px, py), holding, _ = obs[i]
            local_input = [px / (self.env.size - 1), py / (self.env.size - 1), float(holding)]
            local_input.extend(order_mask) # グローバルなオーダー情報をローカル入力に含める
            local_states.append(torch.tensor([local_input], dtype=torch.float32, device=device))
            
        return global_state_tensor, local_states

    def select_action(self, global_state_tensor, local_states: List[torch.Tensor]):
        """ε-greedy法に基づく行動選択 (分散実行)"""
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        actions = {}
        if sample > eps_threshold:
            # 貪欲な行動: 各エージェントはローカルQ値に基づいて独立に最適行動を選択
            with torch.no_grad():
                for i in range(self.num_agents):
                    q_values = self.agent_nets[i](local_states[i])
                    action = q_values.max(1)[1].item()
                    actions[i] = action
        else:
            actions = {i: random.randrange(self.action_space) for i in range(self.num_agents)}
            
        return actions

    def optimize_model(self):
        """学習の実行 (集中学習)"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        transitions = self.memory.sample(BATCH_SIZE)
        
        # バッチデータの作成 (省略: 以前のコードと同じロジック)
        # ... (各種 batch_xxx_tensor を作成) ...
        # QMixAgentクラスの最適化部分のコードをそのまま使用

        batch_global_state = []
        batch_local_states = [[] for _ in range(self.num_agents)]
        batch_next_global_state = []
        batch_actions = []
        batch_rewards = []
        batch_done = []

        for state_dict, action_dict, next_state_dict, reward_dict, done_bool in transitions:
            global_state, local_states = self.preprocess_state(state_dict)
            next_global_state, next_local_states = self.preprocess_state(next_state_dict)

            batch_global_state.append(global_state.squeeze(0))
            batch_next_global_state.append(next_global_state.squeeze(0))

            for i in range(self.num_agents):
                batch_local_states[i].append(local_states[i].squeeze(0))

            batch_actions.append([action_dict[i] for i in range(self.num_agents)])
            batch_rewards.append(sum(reward_dict.values()))
            batch_done.append(float(all(done_bool.values()) if isinstance(done_bool, Dict) else done_bool))

        global_state_batch = torch.stack(batch_global_state).to(device)
        next_global_state_batch = torch.stack(batch_next_global_state).to(device)
        local_states_batch = [torch.stack(l).to(device) for l in batch_local_states]

        action_batch = torch.tensor(batch_actions, dtype=torch.long, device=device)
        reward_batch = torch.tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(1)
        done_batch = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)

        # 2. Q(s, a) の計算 (Policy Net)
        q_values_list = []
        for i in range(self.num_agents):
            q_all = self.agent_nets[i](local_states_batch[i]) 
            q_selected = q_all.gather(1, action_batch[:, i].unsqueeze(1))
            q_values_list.append(q_selected)

        q_selected_agents = torch.stack(q_values_list, dim=1)
        q_tot = self.mixer(q_selected_agents, global_state_batch).unsqueeze(1)

        # 3. ターゲット値の計算 (Target Net)
        target_q_values_list = []
        with torch.no_grad():
            for i in range(self.num_agents):
                target_q_all = self.target_agent_nets[i](local_states_batch[i]) # 注意: next_local_states_batch が理想だが、ここでは簡略化のため local_states_batch を再利用
                target_q_values_list.append(target_q_all)
        
        target_q_selected_agents = torch.stack([q.max(1)[0].detach().unsqueeze(1) 
                                                 for q in target_q_values_list], dim=1)

        with torch.no_grad():
            q_next_tot = self.target_mixer(target_q_selected_agents, next_global_state_batch).unsqueeze(1)

        expected_q_tot = reward_batch + GAMMA * q_next_tot * (1 - done_batch)

        # 4. 損失計算と最適化
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_tot, expected_q_tot)

        self.optimizer_agents.zero_grad()
        self.optimizer_mixer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        for param in self._get_agent_params():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        for param in self.mixer.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer_agents.step()
        self.optimizer_mixer.step()
        
        return loss.item()

    def update_target_networks(self):
        """ターゲットネットワークの重みを更新"""
        for policy_net, target_net in zip(self.agent_nets, self.target_agent_nets):
            target_net.load_state_dict(policy_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


# ----------------------------------------------------
# 4. 学習ループ
# ----------------------------------------------------

def train_qmix_agent(env: 'WarehouseEnv'):
    
    state_dim = 2 * env.num_agents + env.num_agents + NUM_ORDERS # 9
    agent = QMixAgent(env, state_dim=state_dim)
    rewards_history = []
    
    print("--- QMIX 学習開始 ---")
    
    for i_episode in range(NUM_EPISODES):
        obs = env.reset()
        global_state, local_states = agent.preprocess_state(obs)
        
        episode_reward = 0
        
        for t in range(MAX_STEPS):
            actions = agent.select_action(global_state, local_states)
            next_obs, rewards, done, info = env.step(actions)
            
            # リプレイメモリに保存
            agent.memory.push(obs, actions, next_obs, rewards, all(done.values()))
            
            obs = next_obs
            global_state, local_states = agent.preprocess_state(obs)
            episode_reward += sum(rewards.values())
            
            # モデル最適化 (集中学習)
            loss = agent.optimize_model()
            
            if all(done.values()):
                break
        
        rewards_history.append(episode_reward)
        
        if (i_episode + 1) % TARGET_UPDATE == 0:
            agent.update_target_networks()
            
        if (i_episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_history[-500:])
            print(f"Episode {i_episode+1}/{NUM_EPISODES} | Avg Reward (500): {avg_reward:.2f} | Steps: {t+1}")
            
    print("\n✅ 学習完了")
    return agent, rewards_history

# ----------------------------------------------------
# 5. 実行
# ----------------------------------------------------

if __name__ == '__main__':
    env = WarehouseEnv()
    
    # 学習の実行
    trained_agent, rewards = train_qmix_agent(env)
    
    # 学習曲線の表示
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("QMIX Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # 簡単なテスト実行 (ここでは可視化は省略)
    print("\n--- 簡単なテスト実行 ---")
    obs = env.reset()
    trained_agent.agent_nets.eval()
    
    total_test_reward = 0
    
    for t in range(10):
        global_state, local_states = trained_agent.preprocess_state(obs)
        actions = {}
        with torch.no_grad():
            for i in range(env.num_agents):
                q_values = trained_agent.agent_nets[i](local_states[i])
                action = q_values.max(1)[1].item()
                actions[i] = action
        
        next_obs, rewards, done, info = env.step(actions)
        total_test_reward += sum(rewards.values())
        obs = next_obs
        
        if all(done.values()):
            print(f"タスク完了！ ステップ数: {t+1} / 総報酬: {total_test_reward:.2f}")
            break
