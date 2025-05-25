
import numpy as np
from environment import Environment
from agent import TdLambdaAgent
import matplotlib.pyplot as plt

def train(num_episodes=1000, max_steps=100):
    env = Environment()
    agent = TdLambdaAgent(env)
    delta_q = []
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        agent.reset_eligibility_trace()
        q_previous = agent.q_function.copy()
        for step in range(max_steps):
            s_idx = agent.state_to_idx(state, env.maze.shape[1])
            action = agent.select_action(s_idx)
            reward, next_state, done = env.step(state, agent.actions[action])
            rewards.append(reward)
            # 次の状態のインデックスを取得
            s_idx_next = agent.state_to_idx(next_state, env.maze.shape[1])
            agent.update(s_idx, action, reward, s_idx_next, done)

            state = next_state

            if done:
                break
        
        # Q値の変化を記録
        delta_q.append(np.max(np.abs(agent.q_function - q_previous)))
        q_previous = agent.q_function.copy()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards[-100:])}")
            rewards = []

    plot_q_function(delta_q)  # Q値の変化をプロット
    
    agent.save()  # Q関数を保存
    print("Training completed.")


def plot_q_function(delta_q):
    plt.plot(delta_q)
    plt.xlabel("Episode")
    plt.ylabel("Max Q-Value Change")
    plt.title("Q-Value Convergence")
    plt.show()


# エージェントを評価する関数
def evaluate_agent():
    env = Environment()
    agent = TdLambdaAgent(env)
    env = Environment()
    agent = TdLambdaAgent(env)
    num_episodes = 1
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        agent.reset_eligibility_trace()
        while True:
            s_idx = agent.state_to_idx(state, env.maze.shape[1])
            action = agent.select_action(s_idx)
            reward, next_state, done = env.step(state, agent.actions[action])
            # 次の状態のインデックスを取得
            state = next_state

            if done:
                break
    # 関数用に配列の並びを変更
    q_function = agent.q_function.reshape(env.maze.shape[0], env.maze.shape[1], -1)
    draw_optimal_action(q_function, env)


# 最適行動を描画する関数
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
    # train(num_episodes=1000, max_steps=100)
    evaluate_agent()