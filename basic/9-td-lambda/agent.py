
import os
import numpy as np

class TdLambdaAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, lambd=0.9):
        self.env = env
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.lambd = lambd  # λパラメータ
        self.epsilon = 0.9
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # 行動のリスト
        self.q_function = np.zeros((env.maze.shape[0] * env.maze.shape[1], len(self.actions)))  # 状態-行動価値関数の初期化
        self.eligibility_trace = np.zeros((env.maze.shape[0] * env.maze.shape[1], len(self.actions)))  # eligibility trace
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.__path_q = f"{dir_current}/q_function.npy"  # Q関数の保存パス
        self.load()  # Q関数のロード


    # 状態を行動インデックスに変換
    def state_to_idx(self, state, n_cols):
        return state[0] * n_cols + state[1]

    # ε-greedy法で行動を選択
    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.q_function[state_idx])


    def update(self, s_idx, a, r, s_next_idx, done):
        # Q(λ)更新
        td_error = r + self.gamma * np.max(self.q_function[s_next_idx]) * (not done) - self.q_function[s_idx, a]
        
        self.eligibility_trace[s_idx, a] += 1  # eligibility traceを増加

        # 全状態・全行動について一括更新
        self.q_function += self.alpha * td_error * self.eligibility_trace
        # eligibility trace減衰
        self.eligibility_trace *= self.gamma * self.lambd

        if done:
            self.reset_eligibility_trace()  # エピソード終了時はtraceをリセット

        # 行動選択のεを減衰
        self.epsilon *= 0.999

    # 適合度トレースをリセット
    def reset_eligibility_trace(self):
        self.eligibility_trace.fill(0)

    # Q関数を保存する
    def save(self):
        np.save(self.__path_q, self.q_function)

    # Q関数をロードする
    def load(self):
        if os.path.exists(self.__path_q):
            print("Loading Q-function from file")
            self.q_function = np.load(self.__path_q)
        self.reset_eligibility_trace()  # eligibility traceもリセット