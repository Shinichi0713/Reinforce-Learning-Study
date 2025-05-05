# 方策反復
import matplotlib.pyplot as plt
import numpy as np
import agent, environment


# 方策評価→方策改善
# --- 方策反復プログラム ---
# def policy_iteration():
#     env = environment.Environment()
#     agent_instance = agent.Agent(env)
#     iteration = 0
#     while True:
#         iteration += 1
#         value_function = agent_instance.evaluate_pi()
#         is_pi_stable = agent_instance.improve_pi()
#         print(f"Iteration {iteration}:")
#         print("Q function:", value_function)
#         print("pi =", agent_instance.pi)
#         if is_pi_stable:
#             break
#     print("\n最適方策:", agent_instance.pi)
#     print("最適状態価値:", agent_instance.value_function)
#     return agent_instance.pi, agent_instance.value_function

def policy_iteration():
    env = environment.Environment()
    agent_instance = agent.Agent(env)
    iteration = 0

    # 学習の経過を保存するリスト
    pi_history = []

    while True:
        iteration += 1
        value_function = agent_instance.evaluate_pi()
        is_pi_stable = agent_instance.improve_pi()

        # deep copyで保存
        pi_history.append(np.copy(agent_instance.pi))

        print(f"Iteration {iteration}:")
        print("Value function:", value_function)
        print("pi =", agent_instance.pi)
        if is_pi_stable:
            break

    print("\n最適方策:", agent_instance.pi)
    print("最適状態価値:", agent_instance.value_function)
    return pi_history, agent_instance.pi

# 方策の変化（例：各状態で選択される行動の推移）
def visualize_policy(pi_history):
    pi_history_arr = np.array(pi_history)  # (iteration, state)
    plt.figure(figsize=(10, 5))
    for state in range(pi_history_arr.shape[1]):
        plt.plot(pi_history_arr[:, state], label=f"State {state}")
    plt.xlabel("Iteration")
    plt.ylabel("Action")
    plt.title("Policy Evolution")
    plt.legend()
    plt.show()

def visualize_policy_1d(pi):
    # 方策のシンボル
    def get_symbol(val):
        if val == 0.0:
            return "←"
        elif val == 1.0:
            return "→"
        elif val == 0.5:
            return "・"
        else:
            return "?"

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(-0.5, pi.shape[1]-0.5)
    ax.set_ylim(-0.5, pi.shape[0]-0.5)
    ax.set_xticks(range(pi.shape[1]))
    ax.set_yticks(range(pi.shape[0]))
    ax.set_xticklabels([f"列{j}" for j in range(pi.shape[1])])
    ax.set_yticklabels([f"行{i}" for i in range(pi.shape[0])])
    ax.invert_yaxis()

    # 方策を各マスに記載
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            symbol = get_symbol(pi[i, j])
            ax.text(j, i, symbol, fontsize=30, ha='center', va='center')

    ax.set_title("最適方策（←：0, →：1, ・：0.5）")
    plt.grid(True)
    plt.show()

# --- 実行 ---
if __name__ == "__main__":
    pi_history, pi = policy_iteration()

    visualize_policy_1d(pi)

