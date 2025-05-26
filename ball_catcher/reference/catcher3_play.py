
import sys, os
dir_root = '/'.join(os.path.dirname(os.path.abspath(__file__)).replace("\\", '/').split("/")[:-2])
print(dir_root)
sys.path.append(dir_root + '/ple-cited')
import pickle
import numpy as np
from ple.ple import PLE
from ple.games.catcher import Catcher
import time

# 状態の離散化関数（学習時と同じものを使うこと！）
def discretize_state(state, bins):
    player_x = int(state["player_x"] // bins)
    fruit_x = int(state["fruit_x"] // bins)
    fruit_y = int(state["fruit_y"] // bins)
    return (player_x, fruit_x, fruit_y)

# パラメータ（学習時と合わせる）
bins = 20

# Qテーブルの読み込み
dir_current = os.path.dirname(os.path.abspath(__file__))
with open(dir_current + "/catcher_q_table.pkl", "rb") as f:
    Q = pickle.load(f)
    print("Qテーブルの読み込み完了")

# 環境の準備
game = Catcher(width=256, height=256)
env = PLE(game, fps=120, display_screen=True)
env.init()
actions = env.getActionSet()

# 自動プレイ
num_episodes = 100  # 何回プレイするか

for episode in range(num_episodes):
    env.reset_game()
    state = discretize_state(game.getGameState(), bins)
    total_reward = 0
    while not env.game_over():
        # Q値最大の行動を選択
        qvals = [Q.get((state, a), 0) for a in actions]
        action = actions[np.argmax(qvals)]
        reward = env.act(action)
        state = discretize_state(game.getGameState(), bins)
        total_reward += reward
        time.sleep(0.02)
    print(f"Episode {episode+1}: total reward = {total_reward}")

env.close()
