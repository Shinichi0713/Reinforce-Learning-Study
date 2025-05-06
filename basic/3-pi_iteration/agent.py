import numpy as np


class Agent:
    def __init__(self, environment):
        self.actions = ["left", "right"]
        # self.value_function = np.zeros((4, 2))
        self.pi = np.array([[0.5, 0.5] for _ in range(4)])
        self.environment = environment
        self.gamma = 0.9
        self.value_function = np.zeros(4)
        self.q_function = np.zeros((4, len(self.actions)))

    def effect_action(self, action):
        if action == 'left':
            return -1
        else:
            return 1

    def __decide_best_pi(self):
        best_actions = np.argmax(self.q_function, axis=1)
        pi_new = np.zeros_like(self.pi)
        for i, best_action in enumerate(best_actions):
            if max(self.q_function[i]) == 0.5:
                pi_new[i] = [0.5, 0.5]
            elif best_action == 0:
                pi_new[i] = [1, 0]
            else:
                pi_new[i] = [0, 1]
        self.pi = pi_new
        return self.pi

    # その段階での行動価値関数を評価
    def evaluate_pi(self, threshold=1e-5):
        while True:
            delta = 0.0
            for i, status in enumerate(self.environment.status):
                q_new = np.zeros(2)
                for j, action in enumerate(self.actions):
                    effect_action = self.effect_action(action)
                    reward = self.environment.give_reward(status, effect_action)
                    status_next = status + effect_action
                    p = 1           # 状態遷移は行動が決まれば決定的=1
                    q_new[j] += p * reward
                    if status_next >= 1 and status_next < self.value_function.shape[0]:
                        q_new[j] += p * (self.gamma * self.value_function[status_next - 1])
                    # 範囲の外
                    else:
                        q_new[j] += 0
                delta = max(delta, abs(self.value_function[i] - max(q_new)))
                self.value_function[i] = max(q_new)
                self.q_function[i] = q_new
            if delta < threshold:
                break
        return self.value_function
    
    # 方策改善を行う
    def improve_pi(self):
        # 各状態で行動価値を算出する
        pi_old = self.pi.copy()
        for i, status in enumerate(self.environment.status):
            if self.environment.is_terminal(status):
                continue
            
            # 行動価値関数算出
            q_function = {}
            for action in self.actions:
                status_next = status + self.effect_action(action)

                reward = self.environment.give_reward(status, self.effect_action(action))
                q_function[action] = reward + self.gamma * self.value_function[status_next] if status_next < len(self.value_function) else reward
        self.__decide_best_pi()
        if np.array_equal(pi_old, self.pi):
            return True
        else:
            return False
    
