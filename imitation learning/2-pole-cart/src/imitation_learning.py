
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent import Actor, Critic, ReplayMemory, AgentImitation


def collect_experience():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()

    agent_expert = Actor(state_dim, action_dim)
    agent_expert.eval()  # エキスパートエージェントは評価モード

    num_episodes = 500
    reward_history = []
    experience_state = []
    experience_action = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
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

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
        reward_history.append(episode_reward)

    # 経験の保存
    dir_store = os.path.join(os.path.dirname(__file__), 'experience')
    if not os.path.exists(dir_store):
        os.makedirs(dir_store)
    np.save(os.path.join(dir_store, 'experience_state.npy'), np.array(experience_state, dtype=np.float32))
    np.save(os.path.join(dir_store, 'experience_action.npy'), np.array(experience_action, dtype=np.int64))

def train_imitation():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()
    agent_imitation = AgentImitation(state_dim, action_dim)
    agent_optim = torch.optim.Adam(agent_imitation.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_episodes = 500
    reward_history = []
    loss_history = []

    # 経験のロード
    dir_store = os.path.join(os.path.dirname(__file__), 'experience')
    path_state = os.path.join(dir_store, 'experience_state.npy')
    path_action = os.path.join(dir_store, 'experience_action.npy')
    if os.path.exists(path_state) and os.path.exists(path_action):
        experience_state = np.load(path_state)
        experience_action = np.load(path_action)
    else:
        assert False, "Experience data not found. Please run collect_experience() first."

    # 模倣学習のための教師あり学習
    for epoch in range(num_episodes):
        idx = np.random.permutation(len(experience_state))
        states_shuffled = torch.tensor(experience_state[idx]).to(agent_imitation.device)
        actions_shuffled = torch.tensor(experience_action[idx]).to(agent_imitation.device)

        logits = agent_imitation(states_shuffled)
        loss = criterion(logits, actions_shuffled)

        agent_optim.zero_grad()
        loss.backward()
        agent_optim.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


if __name__ == "__main__":
    # collect_experience()
    train_imitation()
