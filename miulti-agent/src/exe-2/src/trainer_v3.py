import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
# %debug
# ----------------------------------------------------
# 0. 環境設定と補助クラス (以前の定義を流用)
# ----------------------------------------------------

# 環境パラメータ (以前の定義に合わせてください)
GRID_SIZE = 10
NUM_AGENTS = 2
NUM_ORDERS = 3
PICKUP_LOCATIONS = [(1, 1), (8, 1), (5, 8)]
DROPOFF_LOCATION = (5, 5)

# QMIX学習パラメータ
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50000        # 減衰をゆっくりにする (ステップ数ベース)
TARGET_UPDATE_INTERVAL = 200 # ターゲットネットワーク更新頻度 (ステップ数)
MEMORY_SIZE = 50000
NUM_EPISODES = 5000      # 実行するエピソード数
MAX_STEPS_PER_EPISODE = 200
LEARNING_FREQ = 4        # 4ステップごとに1回学習

# 環境・エージェントの形状定義
ACTION_SPACE = 5         # 5: 停止, 上, 下, 左, 右
OBS_SHAPE = 2 + 1 + NUM_AGENTS # 位置(2) + 荷物(1) + Agent ID(2) = 5
STATE_SHAPE = (2 + 1) * NUM_AGENTS + NUM_ORDERS # 位置(2) + 荷物(1)) * 2 + 注文(3) = 9

# --- 必須なクラス定義 (以前の回答から流用) ---

# class WarehouseEnv: ... (以前の環境定義をここに配置)
class WarehouseEnv:
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = ACTION_SPACE
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
        # このメソッドの実装は以前の回答と同様に衝突処理、報酬計算などを含む必要があります。
        # 簡略化のため、ここではダミーの結果を返しますが、実際には以前のロジックを配置してください。
        next_obs = self._get_obs()
        rewards = {i: -0.1 for i in range(self.num_agents)}
        done = {i: len(self.remaining_orders) == 0 for i in range(self.num_agents)}
        return next_obs, rewards, done, {}

# class QMixReplayMemory: ... (以前のリプレイメモリ定義をここに配置)
class QMixReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# class MLPAgent: ... (MLPAgentの定義をここに配置)
class MLPAgent(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPAgent, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# class QMixer: ... (QMixerの定義をここに配置)
# 以前の回答から DeepQMixer を QMixer として定義したものを使用
class QMixer(torch.nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim, hypernet_embed_dim):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # ハイパーネットワークの隠れ層サイズ
        hyper_net_hidden = 64

        # --- ハイパーネットワーク W1 (2層構造化) ---
        self.hyper_w1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hyper_net_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_net_hidden, hidden_dim * num_agents)
        )
        self.hyper_b1 = torch.nn.Linear(state_dim, hidden_dim)

        # --- ハイパーネットワーク W2 (2層構造化) ---
        self.hyper_w2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hyper_net_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_net_hidden, hidden_dim)
        )
        self.hyper_b2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hyper_net_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_net_hidden, 1)
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)

        agent_qs = agent_qs.view(batch_size, self.num_agents, 1)
        hidden = torch.bmm(agent_qs.transpose(1, 2), w1)

        b1 = self.hyper_b1(states).view(batch_size, 1, self.hidden_dim)
        hidden = F.elu(hidden + b1)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.hidden_dim, 1)

        q_tot = torch.bmm(hidden, w2)

        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        q_tot = q_tot + b2
        return q_tot.squeeze(-1)

# class IntegratedQMixAgent: ... (前回の修正版クラス定義をここに配置)
# (簡略化のため、init_hiddenなどのRNN関連メソッドは削除)
class IntegratedQMixAgent:
    def __init__(self, env, obs_shape, state_shape, n_actions, lr=5e-4, gamma=0.99, mixing_embed_dim=32, hidden_dim=64, memory_capacity=50000):
        self.env = env
        self.n_agents = env.num_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = QMixReplayMemory(memory_capacity)
        self.agent_net = MLPAgent(obs_shape, hidden_dim, n_actions).to(self.device)
        self.mixer_net = QMixer(self.n_agents, state_shape, mixing_embed_dim, hypernet_embed_dim=hidden_dim).to(self.device)

        self.target_agent_net = MLPAgent(obs_shape, hidden_dim, n_actions).to(self.device)
        self.target_mixer_net = QMixer(self.n_agents, state_shape, mixing_embed_dim, hypernet_embed_dim=hidden_dim).to(self.device)

        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _obs_to_tensor(self, obs: Dict[int, Tuple], is_state: bool = False):
        if is_state:
            state_vec = []
            for i in range(self.n_agents):
                pos_tuple = obs[i][0]
                state_vec.extend([pos_tuple[0] / (GRID_SIZE - 1), pos_tuple[1] / (GRID_SIZE - 1)])
                state_vec.append(1.0 if obs[i][1] else 0.0)
            
            # 残り注文の処理
            remaining_orders_set = set(obs[0][2])
            for order_idx in range(NUM_ORDERS):
                state_vec.append(1.0 if order_idx in remaining_orders_set else 0.0)
            return torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
        
        else:
            tensors = {}
            for i in range(self.n_agents):
                # --- ここも修正：リスト内包表記ではなく明示的なインデックス指定に ---
                pos_tuple = obs[i][0]
                obs_i = [pos_tuple[0] / (GRID_SIZE - 1), pos_tuple[1] / (GRID_SIZE - 1)]
                
                obs_i.append(1.0 if obs[i][1] else 0.0)
                agent_id_vec = [0.0] * self.n_agents
                agent_id_vec[i] = 1.0
                obs_i.extend(agent_id_vec)
                tensors[i] = torch.FloatTensor(obs_i).to(self.device).unsqueeze(0)
            return tensors

    def get_actions(self, obs: Dict[int, Tuple], epsilon: float) -> Dict[int, int]:
        actions = {}
        if random.random() < epsilon:
            for i in range(self.n_agents):
                actions[i] = random.randint(0, self.n_actions - 1)
        else:
            agent_obs_tensors = self._obs_to_tensor(obs, is_state=False)
            with torch.no_grad():
                for i in range(self.n_agents):
                    q_values = self.agent_net(agent_obs_tensors[i])
                    actions[i] = q_values.max(dim=-1)[1].item()
        return actions

    def learn(self, batch, target_update_interval, update_counter):
        current_state = torch.cat([self._obs_to_tensor(t[0], is_state=True) for t in batch], dim=0)
        next_state = torch.cat([self._obs_to_tensor(t[3], is_state=True) for t in batch], dim=0)
        rewards = torch.FloatTensor([sum(t[2].values()) for t in batch]).to(self.device).unsqueeze(1)
        terminated = torch.FloatTensor([t[4] for t in batch]).to(self.device).unsqueeze(1)
        actions_batch = torch.LongTensor([[t[1][i] for i in range(self.n_agents)] for t in batch]).to(self.device)
        obs_batch = [torch.cat([self._obs_to_tensor(t[0], is_state=False)[i] for t in batch], dim=0) for i in range(self.n_agents)]
        next_obs_batch = [torch.cat([self._obs_to_tensor(t[3], is_state=False)[i] for t in batch], dim=0) for i in range(self.n_agents)]

        # 1. 現在のQ_totの計算
        agent_qs = []
        for i in range(self.n_agents):
            q_vals = self.agent_net(obs_batch[i])
            chosen_q = torch.gather(q_vals, dim=1, index=actions_batch[:, i].unsqueeze(1))
            agent_qs.append(chosen_q)

        agent_qs = torch.cat(agent_qs, dim=1)
        q_tot = self.mixer_net(agent_qs, current_state)

        # 2. TDターゲットの計算 (Double DQN 適用)
        target_agent_qs = []
        with torch.no_grad():
            for i in range(self.n_agents):
                next_q_online = self.agent_net(next_obs_batch[i])
                next_action_argmax = next_q_online.max(dim=1)[1].unsqueeze(1)

                target_q_vals = self.target_agent_net(next_obs_batch[i])

                target_max_q = target_q_vals.gather(dim=1, index=next_action_argmax)
                target_agent_qs.append(target_max_q)

            target_agent_qs = torch.cat(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer_net(target_agent_qs, next_state)

        td_target = rewards + self.gamma * target_q_tot * (1 - terminated)

        # 3. 損失の計算と最適化
        loss = F.mse_loss(q_tot, td_target.detach())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.mixer_net.parameters(), 10)

        self.optimizer.step()

        if update_counter % target_update_interval == 0:
            self.update_target_networks()

        return loss.item()

    def update_target_networks(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

# ----------------------------------------------------
# 3. 学習ループ
# ----------------------------------------------------

def run_qmix_training():
    env = WarehouseEnv(GRID_SIZE, NUM_AGENTS)

    # リプレイメモリはここで定義されている
    replay_buffer = QMixReplayMemory(MEMORY_SIZE)

    agent = IntegratedQMixAgent(env, OBS_SHAPE, STATE_SHAPE, ACTION_SPACE, gamma=GAMMA)

    total_steps = 0
    rewards_history = []
    losses_history = []

    print(f"--- QMIX Learning Start --- (Agents: {NUM_AGENTS}, Grid: {GRID_SIZE}x{GRID_SIZE})")

    for i_episode in range(1, NUM_EPISODES + 1):
        obs = env.reset()
        episode_reward = 0
        done_flag = False

        for t in range(MAX_STEPS_PER_EPISODE):
            # 1. ε-greedy法による行動選択
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * total_steps / EPS_DECAY)
            actions = agent.get_actions(obs, epsilon)

            # 2. 環境ステップ
            next_obs, rewards, done, info = env.step(actions)
            terminated_flag = all(done.values())

            # チーム全体の終了フラグ
            terminated_flag = all(done.values())

            # 3. リプレイバッファに保存
            # リプレイメモリに保存されるデータは (obs, actions, next_obs, rewards, terminated) のタプル
            # agent.memory.push(obs, actions, next_obs, rewards, terminated_flag)
            # agent.memory.push(obs, actions, next_obs, rewards, terminated_flag)
            agent.memory.push(obs, actions, rewards, next_obs, terminated_flag)

            obs = next_obs
            episode_reward += sum(rewards.values())
            total_steps += 1

            if terminated_flag:
                done_flag = True

            # 4. 学習ステップ
            if len(agent.memory) > BATCH_SIZE * 5 and total_steps % LEARNING_FREQ == 0:
                batch = agent.memory.sample(BATCH_SIZE)
                loss = agent.learn(batch, TARGET_UPDATE_INTERVAL, total_steps)
                losses_history.append(loss)

            if done_flag:
                break

        rewards_history.append(episode_reward)

        # 進捗報告
        if i_episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_loss = np.mean(losses_history[-100:]) if losses_history else 0.0
            print(f"Epi: {i_episode}/{NUM_EPISODES} | Steps: {t+1} | Total Steps: {total_steps} | Avg R (100): {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.4f}")

    print("\n✅ 学習完了")

    # 5. 結果の可視化
    plt.figure(figsize=(12, 5))
    plt.plot(rewards_history)
    plt.title("QMIX Total Reward per Episode (Moving Avg 100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

# ----------------------------------------------------
# 実行
# ----------------------------------------------------

if __name__ == '__main__':
    run_qmix_training()