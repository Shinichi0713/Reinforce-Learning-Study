
import numpy as np
import matplotlib.pyplot as plt
import environment, agent


# エージェントの訓練
def train_agent(episodes):
    env = environment.Environment()
    agent_instance = agent.Agent(env=env)
    history_value_function = []
    for episode in range(episodes):
        for i, state in enumerate(env.states):
            agent_instance.update_value_function(state)
        # 履歴取得
        history_value_function.append(agent_instance.V.copy())
    print(agent_instance.V)
    return history_value_function

# 価値関数の時系列推移をグラフ化
def display(history_value_function):
    history_value_function = np.array(history_value_function)
    history_value_function = history_value_function.T
    num_rows, num_cols = history_value_function.shape
    fig, axes = plt.subplots(num_rows, 1, figsize=(7, num_rows*3), squeeze=False)
    for i, series in enumerate(history_value_function):
        axes[i, 0].plot(series)
        axes[i, 0].set_xlabel('episode')
        axes[i, 0].set_ylabel('value function')
        axes[i, 0].set_title(f'{i+1} value function shift')
        axes[i, 0].grid(True)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    history_value_function = train_agent(1000)
    display(history_value_function)