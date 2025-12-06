import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import time
# WarehouseEnv は前のコードから引き継がれます

# --- ハイパーパラメータ ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 100
LR_AGENT = 0.0005
LR_MIXER = 0.001
MEMORY_SIZE = 50000
NUM_EPISODES = 10000
MAX_STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 10
NUM_AGENTS = 2
NUM_ORDERS = 3

# 状態空間サイズを計算 (位置10x10 + 荷物有無2 + 残りオーダー数2^3=8ではないが、今回はタプルそのまま)
# 単純化のため、状態は (位置X, 位置Y, 荷物有無, オーダービットマスク) のタプルから変換します。
STATE_DIM = GRID_SIZE * GRID_SIZE + 1 + (2**NUM_ORDERS) # ざっくりとした最大値

# ----------------------------------------------------
# 1. ネットワーク定義
# ----------------------------------------------------

class AgentNet(nn.Module):
    """
    各エージェントの局所Q値を計算するネットワーク
    """
    def __init__(self, input_dim, output_dim):
        super(AgentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class QMixer(nn.Module):
    """
    エージェントのQ値を結合してQ_totを計算するミキシングネットワーク
    モノトニック制約（単調増加）のため、重みは非負にする。
    """
    def __init__(self, num_agents, state_dim, hidden_dim):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # 重みネットワーク (W1: state -> hidden_dim * num_agents)
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        # バイアスネットワーク (B1: state -> hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # 重みネットワーク (W2: state -> hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        # バイアス (B2: state -> 1)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        self.hidden_dim = hidden_dim

    def forward(self, agent_qs, states):
        # agent_qs: (batch_size, num_agents, 1)
        # states: (batch_size, state_dim)
        batch_size = agent_qs.size(0)

        # --- W1 の計算 ---
        # W1 = tanh(hyper_w1(states)) * |hyper_w1| の代わり
        w1 = torch.abs(self.hyper_w1(states)) # モノトニック制約: 重みを非負にする
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim) # (B, N, H)
        
        # --- Q * W1 の計算 ---
        # agent_qs: (B, N, 1) -> (B, N, H) に拡張
        agent_qs = agent_qs.view(batch_size, self.num_agents, 1) 
        
        # 結合 (Q * W1)
        hidden = torch.bmm(agent_qs.transpose(1, 2), w1) # (B, 1, N) x (B, N, H) -> (B, 1, H)

        # --- B1 の計算 ---
        b1 = self.hyper_b1(states) # (B, H)
        b1 = b1.view(batch_size, 1, self.hidden_dim) # (B, 1, H)

        hidden = F.relu(hidden + b1) # (B, 1, H)

        # --- W2 の計算 ---
        w2 = torch.abs(self.hyper_w2(states)) # モノトニック制約
        w2 = w2.view(batch_size, self.hidden_dim, 1) # (B, H, 1)
        
        # --- (Q * W1 + B1) * W2 + B2 の計算 ---
        q_tot = torch.bmm(hidden, w2) # (B, 1, H) x (B, H, 1) -> (B, 1, 1)

        b2 = self.hyper_b2(states) # (B, 1)
        b2 = b2.view(batch_size, 1, 1) # (B, 1, 1)
        
        q_tot = q_tot.view(batch_size, 1) + b2.view(batch_size, 1) # (B, 1)

        return q_tot.squeeze(-1) # (B,)


# ----------------------------------------------------
# 2. Replay Memory (バッチのTensor化処理を強化)
# ----------------------------------------------------

class QMixReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        # state: Dict[int, Tuple]
        # action: Dict[int, int]
        # next_state: Dict[int, Tuple]
        # reward: Dict[int, float]
        # done: bool
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

        # Target Networks
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
        # (位置X, 位置Y, 荷物有無, 残りオーダー数)
        return 2 + 1 + NUM_ORDERS # (10, 10)をそのまま2次元、荷物有無1次元、残りオーダーを3次元のフラグで表現

    def _get_agent_params(self):
        params = []
        for net in self.agent_nets:
            params.extend(list(net.parameters()))
        return params

    def preprocess_state(self, obs: Dict[int, Tuple]):
        """
        環境の状態 (タプル) を学習可能なTensorに変換
        グローバル状態 (ミキサー用) とローカル状態 (エージェントネット用) を作成
        """
        # グローバル状態 (Global State): 全エージェントの位置、荷物有無、残りオーダー
        # (2*N + N + NUM_ORDERS)
        global_state = []
        for i in range(self.num_agents):
            (px, py), holding, _ = obs[i]
            global_state.extend([px / (self.env.size - 1), py / (self.env.size - 1), float(holding)])

        # 残りオーダー (ビットマスクをフラグに変換)
        order_mask = [0.0] * NUM_ORDERS
        remaining_order_ids = obs[0][2] # エージェント0から残りのオーダー情報 (タプル) を取得
        for order_id in remaining_order_ids:
            order_mask[order_id] = 1.0
        global_state.extend(order_mask)
        
        global_state_tensor = torch.tensor([global_state], dtype=torch.float32, device=device)
        
        # ローカル状態 (Agent State):
        local_states = []
        for i in range(self.num_agents):
            (px, py), holding, remaining_orders_tuple = obs[i]
            
            # ローカル入力 (位置, 荷物, オーダーマスク)
            local_input = [px / (self.env.size - 1), py / (self.env.size - 1), float(holding)]
            
            # ローカルもグローバルと同じオーダーマスク情報を持つ
            local_input.extend(order_mask)
            
            local_states.append(torch.tensor([local_input], dtype=torch.float32, device=device))
            
        return global_state_tensor, local_states

    def select_action(self, global_state_tensor, local_states: List[torch.Tensor]):
        """
        ε-greedy法に基づく行動選択
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        actions = {}
        if sample > eps_threshold:
            with torch.no_grad():
                for i in range(self.num_agents):
                    # AgentNetからQ値を計算し、最大の行動を選択
                    q_values = self.agent_nets[i](local_states[i])
                    action = q_values.max(1)[1].item()
                    actions[i] = action
        else:
            # ランダム行動
            actions = {i: random.randrange(self.action_space) for i in range(self.num_agents)}
            
        return actions

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        
        # 1. バッチデータの処理とTensor化
        # QMIXでは、グローバル状態、エージェント行動、エージェント報酬をまとめて処理
        
        batch_global_state = []
        batch_local_states = [] # N x (B, InputDim)
        batch_next_global_state = []
        batch_actions = []
        batch_rewards = []
        batch_done = []
        
        for state_dict, action_dict, next_state_dict, reward_dict, done_bool in transitions:
            # グローバル状態のテンソル化（ここで前処理を再実行）
            global_state, local_states = self.preprocess_state(state_dict)
            next_global_state, next_local_states = self.preprocess_state(next_state_dict)
            
            batch_global_state.append(global_state.squeeze(0))
            batch_next_global_state.append(next_global_state.squeeze(0))
            
            # ローカル状態をエージェントごとにリストに追加
            if not batch_local_states:
                batch_local_states = [[] for _ in range(self.num_agents)]
            for i in range(self.num_agents):
                batch_local_states[i].append(local_states[i].squeeze(0))

            # 行動と報酬
            batch_actions.append([action_dict[i] for i in range(self.num_agents)])
            # QMIXではチーム報酬 (Total reward) を使用
            batch_rewards.append(sum(reward_dict.values()))
            batch_done.append(float(done_bool))

        # Tensorに変換
        global_state_batch = torch.stack(batch_global_state).to(device) # (B, StateDim)
        next_global_state_batch = torch.stack(batch_next_global_state).to(device) # (B, StateDim)
        
        local_states_batch = [torch.stack(l).to(device) for l in batch_local_states] # N x (B, InputDim)
        
        action_batch = torch.tensor(batch_actions, dtype=torch.long, device=device) # (B, N)
        reward_batch = torch.tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(1) # (B, 1)
        done_batch = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1) # (B, 1)

        # 2. Q(s, a) の計算 (Policy Net)
        
        q_values_list = []
        for i in range(self.num_agents):
            # Q値の計算 (B, ActionSpace)
            q_all = self.agent_nets[i](local_states_batch[i]) 
            # 実行された行動に対応するQ値を取得 (B, 1)
            q_selected = q_all.gather(1, action_batch[:, i].unsqueeze(1))
            q_values_list.append(q_selected)

        # 各エージェントのQ値をスタック (B, N, 1)
        q_selected_agents = torch.stack(q_values_list, dim=1)
        # Mixerでチーム全体のQ値を計算 (B, 1)
        q_tot = self.mixer(q_selected_agents, global_state_batch).unsqueeze(1) # (B, 1)

        # 3. ターゲット値の計算 (Target Net)
        
        # Target Agent Netで次状態のQ値を計算
        target_q_values_list = []
        for i in range(self.num_agents):
            with torch.no_grad():
                target_q_all = self.target_agent_nets[i](next_local_states[i])
                target_q_values_list.append(target_q_all)
        
        # **DQNと同様、ターゲットQ値は最大行動を選択 (貪欲)**
        # 各エージェントで最適な行動に対応するQ値を取得 (B, 1)
        target_q_selected_agents = torch.stack([q.max(1)[0].detach().unsqueeze(1) 
                                                for q in target_q_values_list], dim=1) # (B, N, 1)

        # Target Mixerでチーム全体のQ'を計算
        with torch.no_grad():
            q_next_tot = self.target_mixer(target_q_selected_agents, next_global_state_batch).unsqueeze(1) # (B, 1)

        # ターゲット値の計算: R + γ * Q_next_tot * (1 - done)
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
        # Target Agent Nets
        for policy_net, target_net in zip(self.agent_nets, self.target_agent_nets):
            target_net.load_state_dict(policy_net.state_dict())
        # Target Mixer
        self.target_mixer.load_state_dict(self.mixer.state_dict())


# ----------------------------------------------------
# 4. 学習ループ
# ----------------------------------------------------

def train_qmix_agent(env: 'WarehouseEnv'):
    
    # グローバル状態の次元を再計算（QMixAgent.__init__内と同じロジック）
    state_dim = 2 * env.num_agents + env.num_agents + NUM_ORDERS # (位置 + 荷物 + オーダーマスク)
    agent = QMixAgent(env, state_dim=state_dim)
    
    rewards_history = []
    
    print("--- QMIX 学習開始 ---")

    for i_episode in range(NUM_EPISODES):
        obs = env.reset()
        global_state, local_states = agent.preprocess_state(obs)
        
        episode_reward = 0
        
        for t in range(MAX_STEPS):
            # 1. 行動決定
            actions = agent.select_action(global_state, local_states)
            
            # 2. 環境実行
            next_obs, rewards, done, info = env.step(actions)
            
            # 3. 経験の保存
            # next_obsは次の状態のdict
            agent.memory.push(obs, actions, next_obs, rewards, all(done.values()))
            
            # 4. 状態更新と報酬集計
            obs = next_obs
            global_state, local_states = agent.preprocess_state(obs)
            episode_reward += sum(rewards.values())
            
            # 5. モデルの最適化
            loss = agent.optimize_model()
            
            if all(done.values()):
                break
        
        rewards_history.append(episode_reward)
        
        # ターゲットネットワークの定期更新
        if (i_episode + 1) % TARGET_UPDATE == 0:
            agent.update_target_networks()
            
        # ログ出力
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {i_episode+1}/{NUM_EPISODES} | Avg Reward (100): {avg_reward:.2f} | Steps: {t+1}")
            
    print("\n✅ 学習完了")
    return agent, rewards_history


# ----------------------------------------------------
# 5. 実行とテスト
# ----------------------------------------------------

if __name__ == "__main__":
    env = WarehouseEnv(size=GRID_SIZE, num_agents=NUM_AGENTS)
    
    # 学習の実行
    trained_agent, rewards = train_qmix_agent(env)
    
    # 学習曲線の表示
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("QMIX Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # --- 学習済みモデルでのテスト実行 ---
    print("\n--- テスト実行 (可視化) ---")
    obs = env.reset()
    trained_agent.agent_nets.eval() # 評価モード
    
    for t in range(MAX_STEPS):
        global_state, local_states = trained_agent.preprocess_state(obs)

        # ε=0 (貪欲) で行動を選択
        actions = {}
        with torch.no_grad():
            for i in range(env.num_agents):
                q_values = trained_agent.agent_nets[i](local_states[i])
                action = q_values.max(1)[1].item()
                actions[i] = action
        
        next_obs, rewards, done, info = env.step(actions)
        
        # 可視化
        env.render(mode='graphic', sleep_time=0.1) 
        
        obs = next_obs
        
        if all(done.values()):
            print(f"テスト完了！ 全てのタスクを {t+1} ステップで完了しました。")
            plt.pause(2.0)
            break
            
    plt.ioff()
    plt.show()