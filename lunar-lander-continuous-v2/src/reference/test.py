import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")
n_actions = env.action_space.n  # 4
obs_low = env.observation_space.low
obs_high = env.observation_space.high

# 離散化の分割数
n_bins = (6, 6, 6, 6, 6, 6, 2, 2)
bins = [
    np.linspace(obs_low[i], obs_high[i], n_bins[i] - 1)
    for i in range(len(n_bins))
]

def discretize(obs):
    return tuple(
        int(np.digitize(obs[i], bins[i]))
        for i in range(len(obs))
    )

# Qテーブル初期化
q_table = np.zeros(n_bins + (n_actions,))

alpha = 0.1      # 学習率
gamma = 0.99     # 割引率
epsilon = 0.1    # ε-greedy

n_episodes = 1000

for episode in range(n_episodes):
    obs, info = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0

    while not done:
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_obs)

        q_table[state + (action,)] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state + (action,)]
        )

        state = next_state
        total_reward += reward
        done = terminated or truncated

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, total reward: {total_reward}")

env.close()
