


import gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from environment import Environment, MyWalkerWrapper, BoxToHistoryBox
from agent.vit_agent import SacAgent  # ここではSacAgentを仮定

# --- 必要なクラス（あなたの提示コードを想定） ---
# Environment, MyWalkerWrapper, BoxToHistoryBox, SacAgent
# ここでは省略します（上記のコードをそのまま使ってください）

# --- 簡易リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# --- 学習ループ ---
def train_sac_agent(
    num_episodes=3000,
    max_steps=1000,
    batch_size=64,
    start_steps=10000,
    update_after=1000,
    update_every=50
):
    # 環境とラッパー
    env = Environment(is_train=True)
    # 例: 履歴ラッパーを使いたい場合
    # env = BoxToHistoryBox(env.env, h=4)
    obs_dim = env.reset().shape[0]
    action_dim = env.env.action_space.shape[0]

    agent = SacAgent(action_dim)
    replay_buffer = ReplayBuffer()

    returns = []

    total_steps = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        for step in range(max_steps):
            # 行動選択
            if total_steps < start_steps:
                action = env.env.action_space.sample()
            else:
                action = agent.select_action(state)

            # 環境ステップ
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state
            total_steps += 1

            # 学習ステップ
            if total_steps > update_after and len(replay_buffer) > batch_size and total_steps % update_every == 0:
                for _ in range(update_every):
                    agent.update(replay_buffer, batch_size)

            if done:
                break

        returns.append(episode_return)
        print(f"Episode {episode+1} Return: {episode_return}")
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1} Average Return: {np.mean(returns[-10:])}")
            agent.save_models()
    env.close()
    return returns

# --- 実行 ---
if __name__ == "__main__":
    train_sac_agent(num_episodes=100)
