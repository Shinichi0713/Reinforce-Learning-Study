# bipedalwalkerhardcoreの訓練とプレイ
import os
import torch
import numpy as np
from agent import Actor, Critic, ReplayBuffer
from environment import Environment
import torch.nn.functional as F


class Trainer:
    def __init__(self, state_dim, action_dim, max_action, min_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.max_action = max_action
        self.min_action = min_action
        self.replay_buffer = ReplayBuffer(size_max=1000000, batch_size=64)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def save_models(self):
        self.actor.save()
        self.critic.save()

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.actor.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, self.min_action, self.max_action)

    def train_td3(self):
        # 環境の初期化
        env = Environment(is_train=True)
        state_dim, action_dim = env.get_dimensions()
        max_action = float(env.env.action_space.high[0])
        min_action = float(env.env.action_space.low[0])
        episodes = 1000
        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            with torch.no_grad():
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.replay_buffer.add((state, action, reward, next_state, float(done)))
                    state = next_state
                    episode_reward += reward

if __name__ == "__main__":
    trainer = Trainer(state_dim=24, action_dim=4, max_action=1.0, min_action=-1.0)
    trainer.train_td3()
    trainer.save_models()
    print("Training completed.")
    


