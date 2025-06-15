
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result_loss(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Result(loss shift of DDQN)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_ddqn = os.path.join(dir_current, "loss_history_ddqn.png")
    plt.savefig(path_loss_ddqn)

def display_result_reward(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Result(reward shift of DDQN)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_reward_ddqn = os.path.join(dir_current, "reward_history_ddqn.png")
    plt.savefig(path_reward_ddqn)

def read_result(file_path):
    try:
        with open(file_path, 'r') as file:
            string_data = file.read()
            result = ast.literal_eval(string_data)
            return np.array(result)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # テスト用の結果データ
    dir_current = os.path.dirname(os.path.abspath(__file__))
    # path_loss_ddqn = os.path.join(dir_current, "loss_history_ddqn.txt")
    # display_result_loss(read_result(path_loss_ddqn))

    path_reward_ddqn = os.path.join(dir_current, "reward_history_ddqn.txt")
    display_result_reward(read_result(path_reward_ddqn))