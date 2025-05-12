
import os, sys, random
import numpy as np
dir_current = os.path.dirname(os.path.abspath(__file__))
dir_parent = os.path.dirname(dir_current)
sys.path.append(dir_parent)
import environment

class SarsaAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_row, env.n_col, len(self.actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            r, c = state
            return self.actions[np.argmax(self.Q[r, c])]

    def update(self, state, action, reward, next_state, next_action):
        r, c = state
        a = self.actions.index(action)
        nr, nc = next_state
        na = self.actions.index(next_action)
        self.Q[r, c, a] += self.alpha * (reward + self.gamma * self.Q[nr, nc, na] - self.Q[r, c, a])
