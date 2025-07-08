# 環境とリプレイバッファの定義
import gymnasium as gym
from collections import deque
import random
import numpy as np


# --- 環境クラス ---
class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("CarRacing-v3")
        else:
            self.env = gym.make("CarRacing-v3", render_mode="human")
        self.observation, self.info = self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.observation, self.info = self.env.reset()
        return self.observation

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
    

if __name__ == "__main__":
    env = Environment(is_train=False)
    obs = env.reset()
    done = False
    while not done:
        action = env.env.action_space.sample()  # ランダムなアクション
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()