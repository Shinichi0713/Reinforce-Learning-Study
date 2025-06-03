
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Result(loss shift of DDQN)')
    plt.show()

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
    path_loss_ddqn = os.path.join(dir_current, "loss_history_ddqn.txt")
    display_result(read_result(path_loss_ddqn))
