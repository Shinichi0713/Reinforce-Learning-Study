
import numpy as np
import random, os
import environment
from agent import DQNNetwork

# 経験再生バッファ
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_, done))
        else:
            self.buffer[self.position] = (s, a, r, s_, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

