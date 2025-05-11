import numpy as np
from ple import PLE
from ple.games.catcher import Catcher
import random
import pickle

# 離散化のための関数
def discretize_state(state, bins):
    # 状態は dict {"player_x":..., "fruit_x":..., "fruit_y":...}
    # それぞれをbinsで離散化
    player_x = int(state["player_x"] // bins)
    fruit_x = int(state["fruit_x"] // bins)
    fruit_y = int(state["fruit_y"] // bins)
    return (player_x, fruit_x, fruit_y)

# パラメータ
bins = 20  # 離散化の粒度
alpha = 0.1     # 学習率
gamma = 0.99    # 割引率
epsilon = 0.1   # ε-greedy
num_episodes = 1000

# 環境準備
game = Catcher(width=256, height=256)
env = PLE(game, fps=30, display_screen=False)
env.init()
actions = env.getActionSet()  # [左, 何もしない, 右]

# Qテーブル
Q = {}
# Qテーブルの読み込み
with open("catcher_q_table.pkl", "rb") as f:
    Q = pickle.load(f)
    print("Qテーブルの読み込み完了")

for episode in range(num_episodes):
    env.reset_game()
    state = discretize_state(game.getGameState(), bins)
    total_reward = 0

    while not env.game_over():
        # ε-greedyに行動選択
        if random.random() < epsilon or state not in Q:
            action = random.choice(actions)
        else:
            action = actions[np.argmax([Q.get((state, a), 0) for a in actions])]

        reward = env.act(action)
        next_state = discretize_state(game.getGameState(), bins)

        # Qテーブルの初期化
        for a in actions:
            if (state, a) not in Q:
                Q[(state, a)] = 0

        # Q学習更新
        future_q = max([Q.get((next_state, a), 0) for a in actions])
        Q[(state, action)] += alpha * (reward + gamma * future_q - Q[(state, action)])

        state = next_state
        total_reward += reward

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, total reward: {total_reward}")

# Qテーブルを保存
with open("catcher_q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("学習完了！")

# --- 学習済みQテーブルでプレイ ---
env.display_screen = True
for episode in range(3):
    env.reset_game()
    state = discretize_state(game.getGameState(), bins)
    while not env.game_over():
        # greedyに行動選択
        qvals = [Q.get((state, a), 0) for a in actions]
        action = actions[np.argmax(qvals)]
        env.act(action)
        state = discretize_state(game.getGameState(), bins)
# env.close()
