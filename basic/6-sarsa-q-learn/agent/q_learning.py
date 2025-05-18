
import os, random
import numpy as np


class QLearner:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.9):
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.maze.shape[0], env.maze.shape[1], len(self.actions)))
        self.status = [0, 0]    # 状態
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_q = os.path.join(dir_current, "q-learn.npy")
        if os.path.exists(self.path_q):
            print("Loading Q-Learning from file")
            self.Q = np.load(self.path_q)

    # ε-greedy法
    # epsilonの減衰をさせると、今回の課題が解けなくなる
    def choose_action(self, state, is_training=True):
        if is_training and np.random.rand() < self.epsilon:
            self.epsilon *= 0.9999
            return random.choice(self.actions)
        else:
            row, col = state
            self.epsilon *= 0.9999
            return self.actions[np.argmax(self.Q[row, col])]
        

    # 行動価値関数の更新式
    def update(self, state, action, reward, next_state, next_action):
        row, col = state
        a = self.actions.index(action)
        nr, nc = next_state
        max_q_next = np.max(self.Q[nr, nc])     # SARSAとことなるのはここ
        self.Q[row, col, a] += self.alpha * (reward + self.gamma * max_q_next - self.Q[row, col, a])

    # 行動価値関数を保存する
    def save(self):
        print("モデルを保存")
        np.save(self.path_q, self.Q)

if __name__ == "__main__":
    import sys
    dir_current = os.path.dirname(os.path.abspath(__file__))
    dir_parent = os.path.dirname(dir_current)
    sys.path.append(dir_parent)

    import environment
    env = environment.Environment()
    agent = QLearner(env)
    for episode in range(1000):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            reward, state_next, done = env.step(state, action)
            next_action = agent.choose_action(state_next)
            agent.update(state, action, reward, state_next, next_action)
            state = state_next
            action = next_action
            if done:
                break
    # 行動価値関数保存
    agent.save()

