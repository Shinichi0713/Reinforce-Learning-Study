
import numpy as np

class TdLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, lambd=0.9):
        self.env = env
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.lambd = lambd  # λパラメータ
        self.epsilon = 0.9
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # 行動のリスト
        self.q_function = np.zeros((env.maze.shape[0] * env.maze.shape[1], len(self.actions)))  # 状態-行動価値関数の初期化
        self.eligibility_trace = np.zeros((env.maze.shape[0] * env.maze.shape[1], len(self.actions)))  # eligibility trace

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

    # def update(self, state, action, reward, next_state, done):
    #     # TD(λ)更新
    #     state_action = (tuple(state), action)
    #     next_state_action = (tuple(next_state), self.select_action(next_state, epsilon=0.0))
        
    #     # TD誤差の計算
    #     td_target = reward + (0 if done else self.gamma * self.get_state_action_value(*next_state_action))
    #     td_error = td_target - self.get_state_action_value(*state_action)

    #     # 適合度トレースの更新
    #     self.set_eligibility_trace(state, action, self.get_eligibility_trace(state, action) + 1)

    #     # 状態-行動価値関数の更新
    #     for sa in self.state_action_values.keys():
    #         self.set_state_action_value(sa[0], self.actions[sa[1]], self.get_state_action_value(*sa) + 
    #                                     self.alpha * td_error * self.get_eligibility_trace(sa[0], self.actions[sa[1]]))

    #     # 適合度トレースの減衰
    #     # ここを検討する
    #     for sa in self.eligibility_trace.keys():
    #         self.set_eligibility_trace(sa[0], self.actions[sa[1]], self.get_eligibility_trace(sa[0], self.actions[sa[1]]) * self.gamma * self.lambd)

    #     # エピソード終了時に適合度トレースをリセット
    #     if done:
    #         self.reset_eligibility_trace()