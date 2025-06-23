
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result_loss(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Result(loss shift of Imitation Learning)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_imitation = os.path.join(dir_current, "loss_history_imitation.png")
    plt.savefig(path_loss_imitation)

def display_result_reward(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Result(reward shift of Imitation Learning)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_reward_imitation = os.path.join(dir_current, "reward_history_imitation.png")
    plt.savefig(path_reward_imitation)

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
    path_loss = os.path.join(dir_current, "loss_history_imitation.txt")
    # display_result_loss(read_result(path_loss))

    path_reward = os.path.join(dir_current, "reward_history_imitation.txt")
    display_result_reward(read_result(path_reward))