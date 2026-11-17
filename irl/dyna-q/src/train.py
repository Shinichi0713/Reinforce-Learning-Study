import numpy as np
import random
from collections import defaultdict

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # up
            x = max(0, x-1)
        elif action == 1: # down
            x = min(self.size-1, x+1)
        elif action == 2: # left
            y = max(0, y-1)
        elif action == 3: # right
            y = min(self.size-1, y+1)

        next_state = (x, y)
        reward = 1 if next_state == self.goal else -0.1
        done = next_state == self.goal

        self.state = next_state
        return next_state, reward, done

# ------------------------------
# Dyna-Q
# ------------------------------
alpha = 0.1        # 学習率
gamma = 0.95       # 割引率
epsilon = 0.1      # ε-greedy
planning_steps = 20  # モデルからの学習回数

env = GridWorld()
Q = defaultdict(lambda: np.zeros(4))  # 4 actions
model = {}  # モデル： state,action -> (next_state, reward)

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)
    return np.argmax(Q[state])

episodes = 50
for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        # Q-learning update (real experience)
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        # Save model
        model[(state, action)] = (next_state, reward)

        # Planning (simulate experience)
        for _ in range(planning_steps):
            (s, a), (ns, r) = random.choice(list(model.items()))
            Q[s][a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s][a])

        state = next_state

print("学習済み Q値:")
for s, q in Q.items():
    print(s, q)
