
import numpy as np

class TdLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, lambd=0.9):
        self.env = env
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.lambd = lambd  # λパラメータ
        self.state_action_values = {}  # 状態-行動価値関数
        self.eligibility_trace = {}  # 適合度トレース

    def get_state_action_value(self, state, action):
        # 状態-行動価値関数を取得
        return self.state_action_values.get((tuple(state), action), 0.0)
    
    def set_state_action_value(self, state, action, value):
        # 状態-行動価値関数を設定
        self.state_action_values[(tuple(state), action)] = value

    def get_eligibility_trace(self, state, action):
        # 適合度トレースを取得
        return self.eligibility_trace.get((tuple(state), action), 0.0)
    
    def set_eligibility_trace(self, state, action, value):
        # 適合度トレースを設定
        self.eligibility_trace[(tuple(state), action)] = value

    def reset_eligibility_trace(self):
        # 適合度トレースをリセット
        self.eligibility_trace = {}

    def select_action(self, state, epsilon=0.1):
        # ε-greedy法で行動を選択
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.action_space())
        else:
            q_values = [self.get_state_action_value(state, action) for action in range(self.env.action_space())]
            return np.argmax(q_values)
        
    def update(self, state, action, reward, next_state, done):
        # TD(λ)更新
        state_action = (tuple(state), action)
        next_state_action = (tuple(next_state), self.select_action(next_state, epsilon=0.0))
        
        # TD誤差の計算
        td_target = reward + (0 if done else self.gamma * self.get_state_action_value(*next_state_action))
        td_error = td_target - self.get_state_action_value(*state_action)

        # 適合度トレースの更新
        self.set_eligibility_trace(state, action, self.get_eligibility_trace(state, action) + 1)

        # 状態-行動価値関数の更新
        for sa in self.state_action_values.keys():
            self.set_state_action_value(sa[0], sa[1], self.get_state_action_value(*sa) + 
                                        self.alpha * td_error * self.get_eligibility_trace(*sa))

        # 適合度トレースの減衰
        for sa in self.eligibility_trace.keys():
            self.set_eligibility_trace(sa[0], sa[1], self.get_eligibility_trace(*sa) * self.gamma * self.lambd)

        if done:
            self.reset_eligibility_trace()