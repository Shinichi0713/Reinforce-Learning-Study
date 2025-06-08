import numpy as np
from agent import SACAgent, ReplayBuffer
# from environment import make_env, set_seed
from environment import Environment
import matplotlib.pyplot as plt
import os

# --- パラメータ ---
ENV_NAME = "Pendulum-v1"
SEED = 42
EPISODES = 10000
MAX_STEPS = 300
BATCH_SIZE = 256
MEMORY_SIZE = 1000000

def train():
    env = Environment(is_train=True)  # 学習用環境の作成
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]
    max_action = float(env.env.action_space.high[0])
    agent = SACAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    returns = []
    rewards_history = []
    losses_history = []
    
    for episode in range(EPISODES):
        state = env.reset()
        episode_return = 0
        reward_total = 0
        loss_total = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward
            reward_total += reward

            if len(replay_buffer) > BATCH_SIZE:
                loss = agent.update(replay_buffer, BATCH_SIZE)
                loss_total += loss


            if done:
                break
        rewards_history.append(reward_total)
        losses_history.append(loss_total)
        returns.append(episode_return)
        if episode % 10 == 0:
            avg_return = np.mean(returns[-10:])
            print(f"Episode {episode}: Return {episode_return:.2f}, Avg(10) {avg_return:.2f}")
    # ヒストリを画像化
    visualize_graph(rewards_history, losses_history)

    agent.save_networks()
    print("Training completed.")
    env.close()

def visualize_graph(rewards_history, losses_history):
    dir_current = os.path.dirname(os.path.abspath(__file__))
    # 報酬の履歴をグラフ化して保存
    plt.figure(figsize=(10, 4))
    plt.plot(rewards_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_current, "rewards_history.png"))
    plt.close()

    # Lossの履歴をグラフ化して保存
    plt.figure(figsize=(10, 4))
    plt.plot(losses_history, label="Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Episode Losses History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_current, "losses_history.png"))
    plt.close()

def eval():
    env = Environment(is_train=False)  # 学習用環境の作成
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]
    max_action = float(env.env.action_space.high[0])
    agent = SACAgent(state_dim, action_dim, max_action)

    state = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train()
    eval()