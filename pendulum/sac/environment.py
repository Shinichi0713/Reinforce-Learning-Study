# 環境構築する
import gymnasium as gym
import numpy as np

class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("Pendulum-v1")
        else:
            self.env = gym.make("Pendulum-v1", render_mode="human")
        self.reset()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.clip_reward(reward)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def clip_reward(self, reward):
        reward = np.clip(reward, -4, 4)
        return reward + 4

    def give_dimensions(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]

if __name__ == "__main__":
    env = Environment()
    for i in range(1):
        obs, info = env.reset()
        for step in range(20):
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()