# 環境とリプレイバッファの定義
import gymnasium as gym
from collections import deque
import random
import numpy as np


# --- 環境クラス ---
class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("LunarLanderContinuous-v3")
        else:
            self.env = gym.make("LunarLanderContinuous-v3", render_mode="human")
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


# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

