# 訓練、実演のコード
import numpy as np
import random, os
from agent import SACAgent, ReplayBuffer
from environment import Environment

# --- パラメータ ---
EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 256
MEMORY_SIZE = 1000000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2  # エントロピー正則化係数



# --- メインループ ---
def train():
    env = Environment(is_train=True)
    state_dim , action_dim, max_action = env.give_dimensions()
    agent = SACAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    reward_history = []
    loss_critic_history = []
    loss_actor_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_return = 0
        actor_loss_total, critic_loss_total = 0, 0
        count_learn = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward

            if len(replay_buffer) > BATCH_SIZE:
                actor_loss, critic_loss = agent.update(replay_buffer, BATCH_SIZE)
                actor_loss_total += actor_loss
                critic_loss_total += critic_loss
                count_learn += 1

            if done:
                break

        reward_history.append(episode_return / MAX_STEPS)
        loss_actor_history.append(actor_loss_total / (count_learn + 1e-6))
        loss_critic_history.append(critic_loss_total / (count_learn + 1e-6))
        if episode % 10 == 0:
            avg_return = np.mean(reward_history[-10:])
            print(f"Episode {episode}: Return {episode_return:.2f}, Avg(10) {avg_return:.2f}")
    agent.save_networks()

    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(os.path.join(dir_current, "reward_history.txt"), str(reward_history))
    write_log(os.path.join(dir_current, "loss_actor_history.txt"), str(loss_actor_history))
    write_log(os.path.join(dir_current, "loss_critic_history.txt"), str(loss_critic_history))
    print("Training completed.")
    env.close()

def write_log(file_path, data):
    with open(file_path, 'a') as f:
        f.write(data + '\n')

def eval():
    env = Environment(is_train=False)
    state_dim , action_dim, max_action = env.give_dimensions()
    agent = SACAgent(state_dim, action_dim, max_action)

    state, _ = env.reset()
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
    # train()
    eval()
