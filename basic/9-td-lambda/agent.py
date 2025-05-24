


class TdLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, lambd=0.9):
        self.env = env
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.lambd = lambd  # λパラメータ
        self.state_action_values = {}  # 状態-行動価値関数
        self.eligibility_trace = {}  # 適合度トレース
