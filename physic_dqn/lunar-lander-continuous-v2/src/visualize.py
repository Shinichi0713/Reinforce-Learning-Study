
import os
import ast
import matplotlib.pyplot as plt
import numpy as np

def display_result_loss(actor_loss, critic1_loss, critic2_loss):
    steps = np.arange(len(actor_loss))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 第一軸（左）: actor loss
    ax1.plot(steps, actor_loss, color='tab:blue', label='Actor Loss')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Actor Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 第二軸（右）: critic1, critic2 loss
    ax2 = ax1.twinx()
    ax2.plot(steps, critic1_loss, color='tab:red', label='Critic1 Loss')
    ax2.plot(steps, critic2_loss, color='tab:green', label='Critic2 Loss')
    ax2.set_ylabel('Critic Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 凡例をまとめて表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('SAC Loss Curves (Actor: Left Axis, Critic: Right Axis)')
    plt.tight_layout()
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_sac = os.path.join(dir_current, "loss_history_sac.png")
    plt.savefig(path_loss_sac)

def display_result_reward(result):
    # 結果をプロット
    plt.plot(np.arange(len(result)), result)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SAC Loss Curves (Actor: Left Axis, Critic: Right Axis)')
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_reward_ddqn = os.path.join(dir_current, "reward_history_sac.png")
    plt.savefig(path_reward_ddqn)

def read_result(file_path):
    try:
        with open(file_path, 'r') as f:
            data = [float(line.strip()) for line in f if line.strip()]
            return np.array(data)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # テスト用の結果データ
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_loss_sac = os.path.join(dir_current, "actor_losses.txt")
    loss_actor = read_result(path_loss_sac)
    path_loss_critic1 = os.path.join(dir_current, "critic1_losses.txt")
    loss_critic1 = read_result(path_loss_critic1)
    path_loss_critic2 = os.path.join(dir_current, "critic2_losses.txt")
    loss_critic2 = read_result(path_loss_critic2)
    display_result_loss(loss_actor, loss_critic1, loss_critic2)

    # path_reward_sac = os.path.join(dir_current, "episode_rewards.txt")
    # display_result_reward(read_result(path_reward_sac))