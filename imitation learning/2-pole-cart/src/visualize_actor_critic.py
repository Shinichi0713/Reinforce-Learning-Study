
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result_loss(loss_actor, loss_critic):
    fig, ax1 = plt.subplots()           # 1つ目の軸（左）
    ax2 = ax1.twinx()                   # 2つ目の軸（右）
    # 結果をプロット
    ax1.plot(np.arange(len(loss_actor)), loss_actor, label='Actor Loss', color='tab:blue')
    ax2.plot(np.arange(len(loss_critic)), loss_critic, label='Critic Loss', color='tab:orange')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Actor Loss')
    ax2.set_ylabel('Critic Loss')
    ax1.set_title('Training Result(loss shift of Actor and Critic)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.7, 0.77))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7))
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_actor_critic = os.path.join(dir_current, "loss_history_actor-critic.png")
    plt.savefig(path_loss_actor_critic)

def display_result_reward(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Result(reward shift of Actor and Critic)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_reward_actor_critic = os.path.join(dir_current, "reward_history_actor-critic.png")
    plt.savefig(path_reward_actor_critic)

def read_result(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 各行をfloatに変換してリスト化
        array = [float(line.strip()) for line in lines if line.strip()]
        return np.array(array)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # テスト用の結果データ
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_actor = os.path.join(dir_current, "loss_actor_history.txt")
    path_loss_critic = os.path.join(dir_current, "loss_critic_history.txt")
    # display_result_loss(read_result(path_loss_actor), read_result(path_loss_critic))

    path_reward = os.path.join(dir_current, "reward_history.txt")
    display_result_reward(read_result(path_reward))