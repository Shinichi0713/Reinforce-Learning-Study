
import numpy as np
import os
import agent, environment
import matplotlib.pyplot as plt


# Q関数の更新
# def update_value_function(agent, env, gamma=0.99, max_iter=100):
def update_q_function(agent_instance, env, gamma=0.9, max_iter=8):
    q_array = agent_instance.q_function.copy()
    for iteration in range(max_iter):
        for i in range(env.maze.shape[0]):
            for j in range(env.maze.shape[1]):
                state = np.array([i, j])
                for no_action, action in enumerate(agent_instance.actions):
                    move = agent_instance.act_dict[action]
                    reward = env.give_reward(state.tolist(), move)
                    
                    # Q関数の更新
                    q_array[no_action, state[0], state[1]] += reward
                    # 状態遷移確率は行動によって確実に決まった場所になる=1
                    for direc_action, action_tmp in enumerate(agent_instance.actions):
                        move = agent_instance.act_dict[action_tmp]
                        next_state = state + move
                        # 外側のアクションは無効
                        if next_state[0] < 0 or next_state[0] >= env.maze.shape[0] or next_state[1] < 0 or next_state[1] >= env.maze.shape[1]:
                            continue
                        q_array[no_action, state[0], state[1]] += agent_instance.pi(next_state, action_tmp) * q_array[direc_action, next_state[0], next_state[1]] * gamma
                        # agent_instance.set_pos(state)   # 元の状態基準で計算する
    agent_instance.q_function = q_array  # 結果を保存


# 最適行動に赤色のラベル、他には指定したカラーラベルをつける
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color

def draw_optimal_action(q_function, env):
    # 最適行動のインデクスを取得
    optimal_actions = np.argmax(q_function, axis=0)
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
                plt.arrow(i+ 0.5, j+0.5, 0.2, 0, width=0.01,head_width=0.15,\
                    head_length=0.2,color='r')
            elif optimal_actions[x,y] ==1:
                plt.arrow(i+ 0.5, j+0.5, 0, 0.2, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')
            elif optimal_actions[x,y] ==2:
                plt.arrow(i+ 0.5, j+0.5, -0.2, 0, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')
            elif optimal_actions[x,y] ==3:
                plt.arrow(i+ 0.5, j+0.5, 0, -0.2, width=0.01,head_width=0.15,\
                    head_length=0.2, color='r')

    plt.show()

# Q関数の計算
def main():
    env = environment.Environment()
    agent_instance = agent.Agent(env.maze.shape)
    num_iterative = 8
    update_q_function(agent_instance, env, max_iter=num_iterative)
    # 結果をコンソールに表示
    print("Qpi")
    print(agent_instance.q_function)
    np.save(os.path.dirname(__file__) + "/map.npy", agent_instance.q_function)
    draw_optimal_action(agent_instance.q_function, env)
    


if __name__ == "__main__":
    main()