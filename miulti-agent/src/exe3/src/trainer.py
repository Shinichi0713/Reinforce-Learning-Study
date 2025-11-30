import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 前回の環境コードが必要ですので、MultiSensorSearchEnvクラスが定義されている前提で進めます
# (もし未定義の場合は、前の回答の環境コードを先に実行してください)

# --- ハイパーパラメータ ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 100
LR = 0.001
MEMORY_SIZE = 10000
HIDDEN_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ニューラルネットワークの定義 (DQN) ---
class DuelingDQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DuelingDQN, self).__init__()
        # 入力: 2チャンネル (マップ + 自分の位置), 高さh, 幅w
        
        # 畳み込み層: 空間的な特徴（未探索エリアの塊など）を抽出
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 畳み込み後のサイズ計算
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        # 全結合層
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.head = nn.Linear(128, outputs)

    def forward(self, x):
        # x shape: (batch, 2, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.head(x)

# --- 2. 経験再生バッファ (Replay Memory) ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        # PyTorch Tensorとして保存するために変換などの処理を挟んでも良い
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 3. エージェントと学習プロセス ---
class SharedAgent:
    def __init__(self, grid_size, action_space):
        self.grid_size = grid_size
        self.action_space = action_space
        
        # ネットワークの構築
        self.policy_net = DuelingDQN(grid_size, grid_size, action_space).to(device)
        self.target_net = DuelingDQN(grid_size, grid_size, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def preprocess_state(self, obs, agent_id):
        """
        環境からの観測をニューラルネットに入力できるTensor形式(2, H, W)に変換
        Ch0: 共有探索マップ
        Ch1: エージェント自身の位置 (One-hot)
        """
        # マップ情報の取得
        explored_map = obs[agent_id]["map"] # shape (H, W)
        
        # エージェント位置情報の作成
        agent_pos = obs[agent_id]["position"] # (x, y)
        pos_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # numpyは(row, col) = (y, x) なので注意
        # 範囲外チェックは環境側で行われている前提だが念のため
        px, py = agent_pos
        if 0 <= py < self.grid_size and 0 <= px < self.grid_size:
            pos_map[py, px] = 1.0
        
        # チャンネル結合 (2, H, W)
        state = np.stack([explored_map, pos_map], axis=0)
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # バッチ次元追加 (1, 2, H, W)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # 最大のQ値を持つ行動を選択
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # ランダム行動
            return torch.tensor([[random.randrange(self.action_space)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        # バッチデータの整理
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
        
        state_batch = torch.cat(batch_state)
        action_batch = torch.cat(batch_action)
        reward_batch = torch.cat(batch_reward)
        next_state_batch = torch.cat(batch_next_state)
        done_batch = torch.cat(batch_done)

        # Q(s, a) の計算
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s') = max_a Q(s', a) の計算 (Target Network使用)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # 終了状態の場合は 0
        expected_state_action_values = (next_state_values * GAMMA * (1 - done_batch)) + reward_batch

        # Loss計算 (Huber Loss or MSE)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 最適化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング (安定化のため)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# --- メイン学習ループ ---

def train_multi_agent_search():
    # 環境の生成 (前の回答のコードが必要です)
    env = MultiSensorSearchEnv(size=10, num_agents=3) 
    
    # 共有エージェント脳の生成
    agent_brain = SharedAgent(grid_size=10, action_space=5)
    
    num_episodes = 500
    rewards_history = []
    coverage_history = []

    print("--- 学習開始 (DQN) ---")

    for i_episode in range(num_episodes):
        obs = env.reset()
        
        # エージェントごとの初期状態Tensorを作成
        state = {i: agent_brain.preprocess_state(obs, i) for i in range(env.num_agents)}
        
        total_reward = 0
        
        for t in range(100): # Max steps
            actions = {}
            
            # 1. 全エージェントの行動決定
            for i in range(env.num_agents):
                action = agent_brain.select_action(state[i])
                actions[i] = action.item()
            
            # 2. 環境実行
            next_obs, rewards, done, info = env.step(actions)
            
            # 3. 経験の保存と状態更新
            for i in range(env.num_agents):
                reward_tensor = torch.tensor([rewards[i]], device=device)
                next_state_tensor = agent_brain.preprocess_state(next_obs, i)
                done_tensor = torch.tensor([float(done)], device=device)
                
                # メモリに追加
                agent_brain.memory.push(state[i], 
                                      torch.tensor([[actions[i]]], device=device), 
                                      next_state_tensor, 
                                      reward_tensor,
                                      done_tensor)
                
                # 状態更新
                state[i] = next_state_tensor
                total_reward += rewards[i]
            
            # 4. モデルの最適化 (1ステップごとに学習)
            agent_brain.optimize_model()
            
            if done:
                break
        
        # ターゲットネットワークの定期更新
        if i_episode % TARGET_UPDATE == 0:
            agent_brain.update_target_network()
            
        rewards_history.append(total_reward)
        coverage_history.append(info['coverage'])
        
        if (i_episode + 1) % 10 == 0:
            print(f"Episode {i_episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Coverage: {info['coverage']*100:.1f}% | Epsilon: {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * agent_brain.steps_done / EPS_DECAY):.2f}")

    print("学習完了！")
    return agent_brain, rewards_history, coverage_history

# --- 実行と検証 ---
if __name__ == "__main__":
    # 学習の実行
    trained_brain, rewards, coverages = train_multi_agent_search()
    
    # 学習曲線の表示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(coverages)
    plt.title("Map Coverage Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Coverage (0.0 - 1.0)")
    plt.show()

    # --- 学習済みモデルでのテスト実行 ---
    print("\n--- テスト実行 (可視化) ---")
    env = MultiSensorSearchEnv(size=10, num_agents=3)
    obs = env.reset()
    state = {i: trained_brain.preprocess_state(obs, i) for i in range(env.num_agents)}
    
    # 探索率(ε)を0にして、学習した知識のみで動かす
    trained_brain.policy_net.eval() 

    for t in range(100):
        actions = {}
        for i in range(env.num_agents):
            with torch.no_grad():
                # ε-greedyを使わず、最大のQ値を持つ行動を選択
                action = trained_brain.policy_net(state[i]).max(1)[1].item()
            actions[i] = action
            
        next_obs, rewards, done, info = env.step(actions)
        
        # 可視化
        env.render(sleep_time=0.2)
        
        state = {i: trained_brain.preprocess_state(next_obs, i) for i in range(env.num_agents)}
        
        if done:
            print(f"テスト完了！ 最終カバレッジ: {info['coverage']*100:.1f}%")
            plt.pause(2.0)
            break
            
    plt.ioff()
    plt.show()