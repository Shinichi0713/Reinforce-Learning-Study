import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, os
from environment import Environment, ReplayBuffer
from agent import SACAgent



# --- 学習ループ ---
def train_sac():
    env = Environment(is_train=True)
    obs_dim = env.env.observation_space.shape[0]
    act_dim = env.env.action_space.shape[0]
    act_limit = float(env.env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(obs_dim, act_dim, act_limit, device)
    replay_buffer = ReplayBuffer(1000000)
    batch_size = 256
    total_steps = 200000
    start_steps = 10000
    update_after = 1000
    update_every = 50
    episode_rewards = []

    # 追加: ロス推移格納用リスト
    actor_losses = []
    critic1_losses = []
    critic2_losses = []

    state = env.reset()
    episode_reward = 0
    episode_steps = 0

    for t in range(total_steps):
        if t < start_steps:
            action = env.env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        episode_steps += 1

        if done:
            state = env.reset()
            episode_rewards.append(episode_reward)
            print(f"Step: {t}, Episode Reward: {episode_reward:.2f}")
            episode_reward = 0
            episode_steps = 0

        # 学習・ロス記録
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                actor_loss, critic1_loss, critic2_loss = agent.update(replay_buffer, batch_size)
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)

        agent.save_models()

    # --- 追加: ロス・報酬推移をファイル保存 ---
    dir_current = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(f"{dir_current}/actor_losses.txt", np.array(actor_losses))
    np.savetxt(f"{dir_current}/critic1_losses.txt", np.array(critic1_losses))
    np.savetxt(f"{dir_current}/critic2_losses.txt", np.array(critic2_losses))
    np.savetxt(f"{dir_current}/episode_rewards.txt", np.array(episode_rewards))

    env.close()

def eval_sac():
    env = Environment(is_train=False)
    obs_dim = env.env.observation_space.shape[0]
    act_dim = env.env.action_space.shape[0]
    act_limit = float(env.env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(obs_dim, act_dim, act_limit, device)
    dir_current = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(f"{dir_current}/sac_actor.pth"):
        agent.actor.load_state_dict(torch.load(f"{dir_current}/sac_actor.pth"))
    if os.path.exists(f"{dir_current}/sac_critic1.pth"):
        agent.critic1.load_state_dict(torch.load(f"{dir_current}/sac_critic1.pth"))
    if os.path.exists(f"{dir_current}/sac_critic2.pth"):
        agent.critic2.load_state_dict(torch.load(f"{dir_current}/sac_critic2.pth"))

    for i in range(5):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        print(f"Episode Reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_sac()
    # eval_sac()

