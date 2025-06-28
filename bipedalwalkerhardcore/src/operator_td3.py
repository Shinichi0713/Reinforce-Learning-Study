# bipedalwalkerhardcoreの訓練とプレイ
import os
import torch
import numpy as np
from agent import Actor, Critic, ReplayBuffer
from environment import Environment
import torch.nn.functional as F

def train_td3():
    # 環境の初期化
    env = Environment(is_train=True)
    state_dim, action_dim = env.get_dimensions()
    max_action = float(env.env.action_space.high[0])
    min_action = float(env.env.action_space.low[0])

    # ネットワークの初期化
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)

    # オプティマイザの設定
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # リプレイバッファの初期化
    replay_buffer = ReplayBuffer(size_max=1000000, batch_size=64)

    # 訓練ループ
    for episode in range(1000):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = actor(torch.FloatTensor(state)).detach().numpy()
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))

            if replay_buffer.len() > 1000:
                # バッチサンプリング
                batch = replay_buffer.sample()
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # クリティックの更新
                with torch.no_grad():
                    target_actions = actor(next_states)
                    target_q1, target_q2 = critic(next_states, target_actions)
                    target_q = rewards + (1 - dones) * 0.99 * torch.min(target_q1, target_q2)

                current_q1, current_q2 = critic(states, actions)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # アクターの更新
                actor_loss = -critic.Q1(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()


if __name__ == "__main__":
    train_td3()
    print("Training completed.")
    


