
import numpy as np
import os
import agent, environment


# Q関数の更新
# def update_value_function(agent, env, gamma=0.99, max_iter=100):
def update_q_function(agent_instance, env, gamma=0.99, max_iter=8):
    q_array = agent_instance.q_function.copy()
    for iteration in range(max_iter):
        for i in range(env.maze.shape[0]):
            for j in range(env.maze.shape[1]):
                state = np.array([i, j])
                for no_action, action in enumerate(agent_instance.actions):
                    pi = agent_instance.pi(state, action)
                    move = agent_instance.act_dict[action]
                    reward = env.give_reward(state.tolist(), move.tolist())
                    # 遷移先の計算
                    next_state = state + move
                    next_state[0] = np.clip(next_state[0], 0, 4)
                    next_state[1] = np.clip(next_state[1], 0, 4)
                    # Q関数の更新
                    q_array[no_action, state[0], state[1]] += reward
                    # 状態遷移確率は行動によって確実に決まった場所になる=1
                    for direc_action, action_tmp in enumerate(agent_instance.actions):
                        q_array[no_action, state[0], state[1]] += pi * q_array[direc_action, next_state[0], next_state[1]] * gamma
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
                q_array[index, i,j] = update_q_function(agent_instance, env,max_iter=8)
    # 結果をコンソールに表示
    print("Qpi")
    print(q_array)
    np.save(os.path.dirname(__file__) + "/map.npy", q_array)


if __name__ == "__main__":
    main()