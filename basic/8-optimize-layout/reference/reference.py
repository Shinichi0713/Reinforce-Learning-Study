import numpy as np
import random

GRID_SIZE = 5
NUM_EPISODES = 5000
ALPHA = 0.1          # 学習率
GAMMA = 0.9          # 割引率
EPSILON = 0.2        # ε-greedy探索率

# 状態: グリッド上の家具配置（ここでは1個だけ配置する例）
# 行動: 家具をどこに置くか（0〜24の25通り）

# Qテーブル（状態は無視して行動のみで管理：単純化のため）
Q = np.zeros(GRID_SIZE * GRID_SIZE)

def reward_func(action):
    # action: 配置場所(0~24)
    # 例: 中心に近いほど高報酬
    x, y = action // GRID_SIZE, action % GRID_SIZE
    center = (GRID_SIZE - 1) / 2
    dist = abs(x - center) + abs(y - center)
    return -dist  # 中心から遠いほどマイナス

for episode in range(NUM_EPISODES):
    # 1エピソード: 1回家具を置く
    # ε-greedyで行動選択
    if random.random() < EPSILON:
        action = random.randint(0, GRID_SIZE * GRID_SIZE - 1)
    else:
        action = np.argmax(Q)
    reward = reward_func(action)
    # Q値更新（状態遷移はないので単純化）
    Q[action] += ALPHA * (reward - Q[action])

# 結果表示
best_action = np.argmax(Q)
x, y = best_action // GRID_SIZE, best_action % GRID_SIZE
print(f"最適配置位置: ({x}, {y})、期待報酬: {Q[best_action]:.2f}")
