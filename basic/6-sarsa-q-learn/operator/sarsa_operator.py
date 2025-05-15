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


def draw_optimal_action(q_function, env):
    # 最適行動のインデクスを取得
    optimal_actions = np.argmax(q_function, axis=2)
    # グラフフレームを作成
    ax = plt.gca()
    plt.xlim(0, env.maze.shape[0])
    plt.ylim(0, env.maze.shape[1])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    print(optimal_actions)

    for i in range(env.maze.shape[0]):
        for j in range(env.maze.shape[1]):
            # 長方形化
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            ax.add_patch(rect)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # arrow
            if optimal_actions[x,y] ==0:
                plt.arrow(i+ 0.5, j+0.5, -0.2, 0, width=0.01,head_width=0.15,\
                    head_length=0.2,color='r')
            elif optimal_actions[x,y] ==1:
                plt.arrow(i+ 0.5, j+0.5, 0.2, 0, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')
            elif optimal_actions[x,y] ==2:
                plt.arrow(i+ 0.5, j+0.5, 0, -0.2, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')
            elif optimal_actions[x,y] ==3:
                plt.arrow(i+ 0.5, j+0.5, 0, 0.2, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')
    plt.show()

if __name__ == "__main__":
    # train_agent()
    evaluate_agent()

