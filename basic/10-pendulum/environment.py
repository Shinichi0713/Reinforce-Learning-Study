# 環境構築する
import gymnasium as gym
import numpy as np

class Environment:
    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.reset()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.clip_reward(reward)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def clip_reward(self, reward):
        if reward < -1.0:
            return -1.0
        else:
            return 1.0


if __name__ == "__main__":
    env = Environment()
    for i in range(1):
        obs, info = env.reset()
        for step in range(20):
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()