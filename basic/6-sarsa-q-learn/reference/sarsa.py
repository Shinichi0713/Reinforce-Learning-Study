import numpy as np
import random

# 迷路データ
maze = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '#', '#', '.'],
    ['#', '.', '.', 'G', '.']
]

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_TO_DELTA = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(ACTIONS)
    else:
        r, c = state
        return ACTIONS[np.argmax(Q[r, c])]

def train_sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = np.zeros((env.n_row, env.n_col, len(ACTIONS)))
    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            r, c = state
            a = ACTIONS.index(action)
            nr, nc = next_state
            na = ACTIONS.index(next_action)
            Q[r, c, a] += alpha * (reward + gamma * Q[nr, nc, na] - Q[r, c, a])
            state = next_state
            action = next_action
    return Q
