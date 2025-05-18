# Sarsaエージェントを操作するコード
import os, sys
import matplotlib.pyplot as plt
import numpy as np
dir_current = os.path.dirname(os.path.abspath(__file__))
dir_parent = os.path.dirname(dir_current)
sys.path.append(dir_parent)

import environment
from agent import SarsaAgent

# エージェントを訓練する関数
def train_agent():
    env = environment.Environment()
    agent = SarsaAgent(env)
    for episode in range(1000):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            reward, state_next, done = env.step(state, action)
            next_action = agent.choose_action(state_next)
            agent.update(state, action, reward, state_next, next_action)
            state = state_next
            action = next_action
            if done:
                break
    # 行動価値関数保存
    agent.save()

# エージェントを評価する関数
def evaluate_agent():
    env = environment.Environment()
    agent = SarsaAgent(env)
    for episode in range(3):
        state = env.reset()
        action = agent.choose_action(state, is_training=False)
        while True:
            state_next = state + action
            next_action = agent.choose_action(state_next, is_training=False)
            done = env.check_over(state_next)
            state = state_next
            action = next_action
            if done:
                break
    # 関数用に配列の並びを変更
    q_function = agent.Q
    draw_optimal_action(q_function, env)


import matplotlib.pyplot as plt
import numpy as np

def draw_optimal_action(q_function, env):
    # 最適行動のインデクスを取得
    optimal_actions = np.argmax(q_function, axis=2)
    rows, cols = env.maze.shape
    # グラフフレームを作成
    fig, ax = plt.subplots(figsize=(cols, rows))
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xticks(np.arange(cols+1))
    ax.set_yticks(np.arange(rows+1))

    # 色マッピング
    color_map = {
        'S': 'cyan',         # 水色
        'G': 'green',        # 緑
        '#': 'saddlebrown'   # 茶色
    }

    for i in range(rows):
        for j in range(cols):
            cell_value = env.maze[i, j]
            facecolor = color_map.get(cell_value, 'white')
            # 下から上に描画するためにy座標を変換
            y = rows - 1 - i
            # セルの背景色
            rect = plt.Rectangle(xy=(j, y), width=1, height=1, 
                                 fill=True, facecolor=facecolor, edgecolor='black')
            ax.add_patch(rect)
            # 矢印（最適行動の描画）
            action = optimal_actions[i, j]
            if action == 2:   # left
                plt.arrow(j+0.5, y+0.5, -0.2, 0, width=0.01, head_width=0.15,
                          head_length=0.2, color='r')
            elif action == 3: # right
                plt.arrow(j+0.5, y+0.5, 0.2, 0, width=0.01, head_width=0.15,
                          head_length=0.2, color='r')
            elif action == 0: # up
                plt.arrow(j+0.5, y+0.5, 0, 0.2, width=0.01, head_width=0.15,
                          head_length=0.2, color='r')
            elif action == 1: # down
                plt.arrow(j+0.5, y+0.5, 0, -0.2, width=0.01, head_width=0.15,
                          head_length=0.2, color='r')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    # train_agent()
    evaluate_agent()

