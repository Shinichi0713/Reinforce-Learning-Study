
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent import Actor, Critic

def train():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99
    num_episodes = 2400
    reward_history = []
    # 記録
    reward_history = []
    loss_actor_history = []
    loss_critic_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = actor(state_tensor).cpu().squeeze(0).numpy()
            action = np.random.choice(action_dim, p=probs)
            next_state, reward, done = env.step(action)

            # 記録
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            episode_reward += reward

        # 学習（エピソード終了後に一括）
        states_tensor = torch.FloatTensor(states).to(actor.device)
        actions_tensor = torch.LongTensor(actions).to(actor.device)
        rewards_tensor = torch.FloatTensor(rewards).to(actor.device)
        next_states_tensor = torch.FloatTensor(next_states).to(actor.device)
        dones_tensor = torch.FloatTensor(dones).to(actor.device)

        # Criticのターゲット（1-step TD）
        with torch.no_grad():
            next_state_values = critic(next_states_tensor)
            targets = rewards_tensor + gamma * next_state_values * (1 - dones_tensor)

        values = critic(states_tensor)
        critic_loss = nn.MSELoss()(values, targets)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Advantage計算
        advantage = (targets - values).detach()

        # Actorの損失
        probs = actor(states_tensor)
        log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).clamp(min=1e-8))
        actor_loss = -(log_probs.squeeze() * advantage).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, reward: {episode_reward}, avg(10): {np.mean(reward_history[-10:]):.1f}")
            actor.save_nn()
            critic.save_nn()
        loss_actor_history.append(actor_loss.item())
        loss_critic_history.append(critic_loss.item())
        reward_history.append(episode_reward)

    # 結果の保存
    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(f"{dir_current}/reward_history.txt", reward_history)
    write_log(f"{dir_current}/loss_actor_history.txt", loss_actor_history)
    write_log(f"{dir_current}/loss_critic_history.txt", loss_critic_history)
    env.close()


# ログファイルに書き込む関数
def write_log(path, data):
    with open(path, 'a') as f:
        for d in data:
            f.write(str(d) + '\n')

# 評価
def evaluate():
    env = Env(is_train=False)
    state_dim, action_dim = env.get_dims()
    actor = Actor(state_dim, action_dim)
    actor.eval()
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            action_prob = actor(state_tensor)[0]
        action = np.random.choice(env.env.action_space.n, p=action_prob.detach().cpu().numpy())
        state, reward, done = env.step(action)
        total_reward += reward
        env.env.render()
        time.sleep(0.01)

    print(f"Total reward in evaluation: {total_reward}")
    env.env.close()


if __name__ == "__main__":
    print("start dqn pole problem")
    # train()
    evaluate()
