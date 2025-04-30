
import numpy as np
import agent, environment


# Q関数の更新
# def update_value_function(agent, env, gamma=0.99, max_iter=100):
def update_q_function(agent_instance, env, gamma=0.99, max_iter=8):
    q_array = agent_instance.q_function.copy()
    for iteration in range(max_iter):
        for i in range(env.maze.shape[0]):
            for j in range(env.maze.shape[1]):
                state = np.array([i, j])
                for action in agent_instance.actions:
                    pi = agent_instance.pi(state, action)
                    move = agent_instance.act_dict[action]
                    reward = env.give_reward(state.tolist(), move.tolist())
                    # 遷移先の計算
                    next_state = state + move
                    next_state[0] = np.clip(next_state[0], 0, 4)
                    next_state[1] = np.clip(next_state[1], 0, 4)
                    # Q関数の更新
                    q_array[action][state[0]][state[1]] += pi * q_array[:, next_state[0], next_state[1]] * gamma
                    agent_instance.set_pos(state)
    agent_instance.q_function = q_array  # 結果を保存


# Q関数の計算
def main():
    env = environment.Environment()
    agent_instance = agent.Agent(env.maze.shape)
    num_iterative = 8
    q_array = np.zeros((len(agent_instance.actions), 5,5))
    for index, action in enumerate(agent_instance.actions):
        print("index: %d" % index)
        for i in range(5):
            for j in range(5):
                q_array[index, i,j] = agent_instance.Q_pi([i,j],action, 1, 0, num_iterative)
    # 結果をコンソールに表示
    print("Qpi")
    print(q_array)


if __name__ == "__main__":
    main()