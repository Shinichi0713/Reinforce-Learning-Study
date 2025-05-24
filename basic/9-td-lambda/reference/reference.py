import numpy as np
import random

# --- 1. 環境定義 ---

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows = len(maze)
        self.n_cols = len(maze[0])
        self.start_pos = self._find_pos('S')
        self.goal_pos = self._find_pos('G')
        self.reset()

    def _find_pos(self, c):
        for r in range(self.n_rows):
            for col in range(self.n_cols):
                if self.maze[r][col] == c:
                    return (r, col)
        return None

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        # 上下左右
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        next_r = self.agent_pos[0] + moves[action][0]
        next_c = self.agent_pos[1] + moves[action][1]
        # 範囲内かつ壁でないなら移動
        if 0 <= next_r < self.n_rows and 0 <= next_c < self.n_cols and self.maze[next_r][next_c] != '#':
            self.agent_pos = (next_r, next_c)
        # 報酬
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 1.0, True  # ゴールで報酬1
        else:
            return self.agent_pos, -0.04, False  # 通常は小さな負の報酬

    def state_space(self):
        return self.n_rows * self.n_cols

    def action_space(self):
        return 4  # 上下左右

# --- 2. エージェント（Q(λ)） ---

class QLambdaAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, lam=0.9, epsilon=0.1):
        self.q = np.zeros((state_size, action_size))
        self.e = np.zeros((state_size, action_size))  # eligibility trace
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

    def state_to_idx(self, state, n_cols):
        return state[0] * n_cols + state[1]

    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.q[state_idx])

    def update(self, s_idx, a, r, s_next_idx, a_next, done):
        # Q(λ)更新
        td_error = r + self.gamma * np.max(self.q[s_next_idx]) * (not done) - self.q[s_idx, a]
        self.e[s_idx, a] += 1  # eligibility traceを増加

        # 全状態・全行動について一括更新
        self.q += self.alpha * td_error * self.e
        # eligibility trace減衰
        self.e *= self.gamma * self.lam

        if done:
            self.e *= 0  # 終了時はtraceリセット

# --- 3. 訓練コード ---

maze = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '#', '#', '.'],
    ['#', '.', '.', 'G', '.']
]

env = MazeEnv(maze)
agent = QLambdaAgent(state_size=env.state_space(), action_size=env.action_space())

n_episodes = 500

for episode in range(n_episodes):
    state = env.reset()
    agent.e *= 0  # eligibility traceリセット
    done = False
    steps = 0

    while not done and steps < 100:
        s_idx = agent.state_to_idx(state, env.n_cols)
        action = agent.select_action(s_idx)
        next_state, reward, done = env.step(action)
        s_next_idx = agent.state_to_idx(next_state, env.n_cols)
        agent.update(s_idx, action, reward, s_next_idx, None, done)
        state = next_state
        steps += 1

    if (episode+1) % 50 == 0:
        print(f"Episode {episode+1} finished in {steps} steps.")

# --- 学習後の方策表示 ---
print("\nLearned Policy (0:Up, 1:Down, 2:Left, 3:Right):")
for r in range(env.n_rows):
    row = ''
    for c in range(env.n_cols):
        if maze[r][c] == '#':
            row += ' # '
        elif maze[r][c] == 'G':
            row += ' G '
        elif maze[r][c] == 'S':
            row += ' S '
        else:
            idx = agent.state_to_idx((r, c), env.n_cols)
            best_action = np.argmax(agent.q[idx])
            row += f' {best_action} '
    print(row)
