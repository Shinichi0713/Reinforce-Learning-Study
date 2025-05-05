import numpy as np


class Agent:
    def __init__(self, environment):
        self.actions = ["left", "right"]
        # self.value_function = np.zeros((4, 2))
        self.pi = np.array([[0.5, 0.5] for _ in range(4)])
        self.environment = environment
        self.gamma = 0.99
        self.q_function = np.zeros((4, len(self.actions)))

    def effect_action(self, action):
        if action == 'left':
            return -1
        else:
            return 1

    def __decide_best_pi(self):
        best_actions = np.argmax(self.q_function, axis=1)
        for i, best_action in enumerate(best_actions):
            self.pi[i] = [1 if action == best_action else 0 for action in self.actions]
        return self.pi

    # その段階での行動価値関数を評価・更新
    def update_pi(self, threshold=1e-5):
        while True:
            delta = 0.0
            V_old = np.max(self.q_function, axis=1)
            # 行動価値関数更新
            for i, status in enumerate(self.environment.status):
                
                for j, action in enumerate(self.actions):
                    effect_action = self.effect_action(action)
                    reward = self.environment.give_reward(status, effect_action)
                    status_next = status + effect_action
                    p = 1           # 状態遷移は行動が決まれば決定的=1
                    self.q_function[i, j] += p * reward
                    if status_next >= 1 and status_next < self.value_function.shape[0]:
                        self.q_function[i, j] += p * (self.gamma * self.value_function[status_next - 1])
                    # 範囲の外
                    else:
                        self.q_function[i, j] += 0
            # 方策更新
            self.__decide_best_pi()
            delta = np.max(np.abs(V_old - np.max(self.q_function, axis=1)))
            if delta < threshold:
                break
        return self.pi
    

if __name__ == "__main__":
    import environment
    environment = environment.Environment()
    agent = Agent(environment)
    agent.update_pi()