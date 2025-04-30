# 環境とエージェントを使って状態価値関数を更新する
import numpy as np
import matplotlib.pyplot as plt
import environment, agent

# 状態価値関数を更新する
def update_value_function(agent, env, gamma=0.99, max_iter=100):
    """
    方策評価（policy evaluation）を繰り返して、状態価値関数を計算する
    gamma: 割引率
    threshold: 収束判定用の閾値
    max_iter: 最大繰り返し回数
    """
    V = agent.value_function.copy()
    for iteration in range(max_iter):
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                new_v = 0
                state = np.array([i, j])
                for action in agent.actions:
                    prob = agent.pi(state, action)
                    move = agent.act_dict[action]
                    reward = env.give_reward(state.tolist(), move.tolist())
                    # 遷移先の計算
                    next_state = state + move
                    next_state[0] = np.clip(next_state[0], 0, 4)
                    next_state[1] = np.clip(next_state[1], 0, 4)
                    new_v += prob * (reward + gamma * V[next_state[0], next_state[1]])
                V[i, j] = new_v
    agent.value_function = V  # 結果を保存

def display_value_function(value_function):
    plt.figure(figsize=(8, 8))
    plt.title("State Value Function Heatmap")
    plt.xlabel("y")
    plt.ylabel("x")
    im = plt.imshow(value_function, cmap='coolwarm', origin='upper')
    plt.colorbar(im, label='Value')
    plt.scatter([0], [0], color='green', label='Start (S)')   # 始点
    plt.scatter([9], [8], color='red', label='End (E)')       # 終点
    plt.legend(loc='lower right')
    plt.show()

def main():
    env = environment.Environment()
    agent_instance = agent.Agent(env.maze.shape)
    update_value_function(agent_instance, env)
    print("Updated value function:", agent_instance.value_function)
    display_value_function(agent_instance.value_function)
    

if __name__ == "__main__":
    main()