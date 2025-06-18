
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result_loss(loss_actor):
    # 結果をプロット
    plt.plot(np.arange(len(loss_actor)), loss_actor, label='agent loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Result(loss shift of PointerNet)')
    plt.legend()
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_pointer_nn = os.path.join(dir_current, "loss_history_pointer_nn.png")
    plt.savefig(path_loss_pointer_nn)

def display_result_reward(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Result(reward shift of PointerNet)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_reward_pointer_nn = os.path.join(dir_current, "reward_history_pointer_nn.png")
    plt.savefig(path_reward_pointer_nn)

def read_result(file_path):
    try:
        lists = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 空行はスキップ
                    try:
                        values = ast.literal_eval(line)
                        lists.append(values)
                    except Exception as e:
                        print(f"Error parsing line: {e}")
        return np.array(lists[0])
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # テスト用の結果データ
    dir_current = os.path.dirname(os.path.abspath(__file__))
    # path_loss_history = os.path.join(dir_current, "loss_history.txt")
    # display_result_loss(read_result(path_loss_history))

    path_reward = os.path.join(dir_current, "reward_history.txt")
    display_result_reward(read_result(path_reward))