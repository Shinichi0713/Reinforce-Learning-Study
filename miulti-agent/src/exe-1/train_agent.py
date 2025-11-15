import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. Q-Network モデルの定義 ---
class QNetwork(nn.Module):
    """じゃんけんゲームのQ値を近似するシンプルなネットワーク"""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 状態は単一なので、入力サイズは1。実際にはバイアスとして機能
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. 環境 (じゃんけんゲーム) の定義 ---
def get_reward(action1, action2):
    """
    じゃんけんのルールに基づき報酬を計算する (ゼロサム)
    アクション: 0=グー, 1=チョキ, 2=パー
    """
    # 勝利条件: (0, 1) -> グーがチョキに勝つ, (1, 2) -> チョキがパーに勝つ, (2, 0) -> パーがグーに勝つ
    # 勝敗判定: action1 - action2 の結果
    # 0: 引き分け
    # 1 or -2: Agent 1 の勝ち
    # -1 or 2: Agent 1 の負け (Agent 2 の勝ち)
    
    diff = action1 - action2
    
    if diff == 0:  # 引き分け
        return 0, 0
    elif diff in [1, -2]:  # Agent 1 の勝ち
        return 1, -1
    else:  # Agent 2 の勝ち
        return -1, 1

# --- 3. ハイパーパラメータ ---
STATE_SIZE = 1      # 状態の数 (S0 のみ)
ACTION_SIZE = 3     # 行動の数 (グー, チョキ, パー)
LR = 0.01           # 学習率 (Learning Rate)
GAMMA = 0.99        # 割引率 (Discount Factor) - シングルステップではあまり意味がないが、形式的に設定
EPSILON_START = 1.0 # Epsilon-greedy の初期値
EPSILON_END = 0.01  # Epsilon-greedy の最終値
EPSILON_DECAY = 5000 # Epsilon の減衰ステップ数
NUM_EPISODES = 20000 # エピソード数 (じゃんけんの試行回数)

# --- 4. エージェントの初期化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent 1
agent1_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
agent1_optimizer = optim.Adam(agent1_net.parameters(), lr=LR)

# Agent 2
agent2_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
agent2_optimizer = optim.Adam(agent2_net.parameters(), lr=LR)

# 状態 S0 の表現 (ダミー入力)
# 状態が一つなので、常に [1.0] というベクトルをネットワークに入力します。
STATE_TENSOR = torch.tensor([1.0], dtype=torch.float32).to(device)

# --- 5. Epsilon-greedy の関数 ---
def select_action(net, epsilon, state):
    if random.random() < epsilon:
        # 探索 (ランダムな行動)
        return random.randrange(ACTION_SIZE)
    else:
        # 活用 (Q値が最大の行動)
        with torch.no_grad():
            q_values = net(state)
            return q_values.argmax().item()

# --- 6. 学習の実行 ---
reward_history = []

for episode in range(NUM_EPISODES):
    # Epsilon の計算 (線形減衰)
    epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY))
    
    # 1. 行動の選択
    action1 = select_action(agent1_net, epsilon, STATE_TENSOR)
    action2 = select_action(agent2_net, epsilon, STATE_TENSOR)
    
    # 2. 環境からの報酬取得
    reward1, reward2 = get_reward(action1, action2)
    
    # --- 3. Q値の更新 (Agent 1) ---
    agent1_optimizer.zero_grad()
    
    # 現在のQ値を取得 Q(s, a1)
    q_values1 = agent1_net(STATE_TENSOR)
    q_a1 = q_values1[action1]
    
    # 目標値 (Target): R + gamma * max(Q'(s', a'))
    # このゲームはシングルステップで終了するため、次状態の最大Q値 max(Q'(s', a')) は 0 とします (GAMMA * 0)。
    target_q1 = reward1
    
    # 損失 (TD Error)
    loss1 = nn.MSELoss()(q_a1, torch.tensor(target_q1).to(device))
    
    # バックプロパゲーションと最適化
    loss1.backward()
    agent1_optimizer.step()
    
    # --- 4. Q値の更新 (Agent 2) ---
    agent2_optimizer.zero_grad()
    
    # 現在のQ値を取得 Q(s, a2)
    q_values2 = agent2_net(STATE_TENSOR)
    q_a2 = q_values2[action2]
    
    # 目標値 (Target)
    target_q2 = reward2
    
    # 損失 (TD Error)
    loss2 = nn.MSELoss()(q_a2, torch.tensor(target_q2).to(device))
    
    # バックプロパゲーションと最適化
    loss2.backward()
    agent2_optimizer.step()

    # 履歴の記録
    reward_history.append((reward1, reward2))
    
    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.4f}, Avg R1: {np.mean([r[0] for r in reward_history[-1000:]]):.3f}")

# --- 7. 学習結果の確認 ---
print("\n--- 学習後の最終 Q値 ---")

# Agent 1
with torch.no_grad():
    q1_final = agent1_net(STATE_TENSOR)
    print(f"Agent 1 Q値 (グー/チョキ/パー): {q1_final.cpu().numpy()}")
    # 最適なじゃんけんは「混合戦略」であるため、Q値はすべて0に近づくのが理論上の理想です。

# Agent 2
with torch.no_grad():
    q2_final = agent2_net(STATE_TENSOR)
    print(f"Agent 2 Q値 (グー/チョキ/パー): {q2_final.cpu().numpy()}")