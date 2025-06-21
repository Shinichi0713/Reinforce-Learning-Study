
import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent import Actor, Critic, ReplayMemory

def train():
    env = Env(is_train=True)
    actor = Actor()
    critic = Critic()
    replay_memory = ReplayMemory()
    action_size = env.env.action_space.n
    state_size = env.env.observation_space.shape[0]

    # 学習の設定
    num_episodes = 1000
    batch_size = 64
    gamma = 0.99  # 割引率
    actor.train()
    critic.train()
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001)
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        loss_actor_total = 0.0
        loss_critic_total = 0.0
        count_train = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
                action_prob = actor(state_tensor)[0]
            action = np.random.choice(action_size, p=action_prob.detach().cpu().numpy())
            state_next, reward, done = env.step(action)
            replay_memory.push((state, action, reward, state_next, done))

            state = state_next
            episode_reward += reward

            if len(replay_memory) > batch_size:
                transitions = replay_memory.sample(batch_size)
                batch = list(zip(*transitions))
                state_batch = torch.from_numpy(np.array(batch[0])).float().to(actor.device)
                action_batch = torch.tensor(batch[1], dtype=torch.int64).to(actor.device)
                reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(actor.device)
                state_next_batch = torch.from_numpy(np.array(batch[3])).float().to(actor.device)
                done_batch = torch.tensor(batch[4], dtype=torch.float32).to(actor.device)

                # Criticの更新
                with torch.no_grad():
                    next_action_prob = actor(state_next_batch)
                    next_action = next_action_prob.argmax(dim=1)
                    next_action_onehot = torch.nn.functional.one_hot(next_action, num_classes=action_size).float()
                    target_q = critic(state_next_batch, next_action_onehot).detach().squeeze(-1)
                    target = reward_batch + gamma * target_q * (1 - done_batch)

                actions_onehot = torch.nn.functional.one_hot(action_batch, action_size).float()
                q_values = critic(state_batch, actions_onehot).squeeze(-1)
                critic_loss = nn.MSELoss()(q_values, target)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                # Actorの更新
                optimizer_actor.zero_grad()
                action_prob = actor(state_batch)
                action_sampled = torch.multinomial(action_prob, num_samples=1).squeeze(1)
                action_sampled_onehot = torch.nn.functional.one_hot(action_sampled, num_classes=action_size).float()
                actor_loss = -critic(state_batch, action_sampled_onehot).mean()
                actor_loss.backward()
                optimizer_actor.step()

                loss_actor_total += actor_loss.item()
                loss_critic_total += critic_loss.item()
                count_train += 1

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: reward {episode_reward} , actor loss {loss_actor_total / max(1, count_train):.4f}, critic loss {loss_critic_total / max(1, count_train):.4f}")
            actor.save_nn()
            critic.save_nn()
    env.env.close()



if __name__ == "__main__":
    print("start dqn pole problem")
    train()
