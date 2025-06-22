
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent import Actor, Critic, ReplayMemory, AgentImitation


def train_imitation():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()

    agent_expert = Actor(state_dim, action_dim)
    agent_expert.eval()  # エキスパートエージェントは評価モード
    agent_imitation = AgentImitation(state_dim, action_dim)
    agent_optim = torch.optim.Adam(agent_imitation.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_episodes = 500
    reward_history = []
    loss_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        experience_state = []
        experience_action = []
        # 倒れるまでアクション
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs_action = agent_expert(state_tensor).cpu().squeeze(0).numpy()

            action = np.random.choice(action_dim, p=probs_action)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            # 経験の蓄積
            experience_state.append(state)
            experience_action.append(action)
            state = next_state

        experience_state = np.array(experience_state, dtype=np.float32)
        experience_action = np.array(experience_action, dtype=np.int64)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Loss: {loss.item()}")

    # 模倣学習のための教師あり学習
    for epoch in range(20):
        idx = np.random.permutation(len(experience_state))
        states_shuffled = torch.tensor(experience_state[idx])
        actions_shuffled = torch.tensor(experience_action[idx])

        logits = agent_imitation(states_shuffled)
        loss = criterion(logits, actions_shuffled)

        agent_optim.zero_grad()
        loss.backward()
        agent_optim.step()

        loss_history.append(loss.item())
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


if __name__ == "__main__":
    train_imitation()
