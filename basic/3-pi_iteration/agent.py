import numpy as np


class Agent:
    def __init__(self, environment):
        self.actions = ["left", "right"]
        # self.value_function = np.zeros((4, 2))
        self.pi = np.array([[0.5, 0.5] for _ in range(4)])
        self.environment = environment
        self.gamma = 0.99
        self.value_function = np.zeros(4)

    def effect_action(self, action):
        if action == 'left':
            return -1
        else:
            return 1

    # その段階での行動価値関数を評価
    def evaluate_pi(self, threthold=1e-5):
        while True:
            delta = 0.0
            for i, status in enumerate(self.environment.status):
                v_new = 0.0
                for j, action in enumerate(self.actions):
                    effect_action = self.effect_action(action)
                    reward = self.environment.give_reward(status, effect_action)
                    pi = self.pi[i, j]
                    status_next = status + effect_action
                    p = 1           # 状態遷移は行動が決まれば決定的=1
                    v_new += pi * p * reward
                    if status_next >= 1 and status_next < self.value_function.shape[0]:
                        v_new += pi * p * (self.gamma * self.value_function[status_next - 1])
                    # 範囲の外
                    else:
                        v_new += 0
                delta = max(delta, abs(self.value_function[i] - v_new))
                self.value_function[i] = v_new
            if delta < threthold:
                break
        return self.value_function
    
    # 方策改善を行う
    def improve_pi(self):
        # 各状態で行動価値を算出する
        for i, status in enumerate(self.environment.status):
            if self.environment.is_terminal(status):
                continue
            pi_old = self.pi.copy()
            # 行動価値関数算出
            q_function = {}
            for action in self.actions:
                status_next = status + self.effect_action(action)

                reward = self.environment.give_reward(status, self.effect_action(action))
                q_function[action] = reward + self.gamma * self.value_function[status_next] if status_next < len(self.value_function) else reward
            best_action = max(q_function, key=q_function.get)
            self.pi[i] = [1 if action == best_action else 0 for action in self.actions]
        if np.array_equal(pi_old, self.pi):
            return True
        else:
            return False
    
if __name__ == "__main__":
    import environment
    env = environment.Environment()
    agent = Agent(env)
    index_action = agent.effect_action("left")
    q_function = agent.evaluate_q_function()
    print("Q function:", q_function)
    agent.improve_q_function()
    print(agent.pi)
