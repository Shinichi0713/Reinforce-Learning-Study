import numpy as np


class Agent:
    def __init__(self, environment):
        self.actions = ["left", "right"]
        self.value_function = np.zeros(4)
        self.pi = np.array([[0.5, 0.5] for _ in range(4)])
        self.environment = environment
        self.gamma = 0.9
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

    # その段階での行動価値関数を評価・更新
    def update_pi(self, threshold=1e-5):
        history_value_function = []
        while True:
            delta = 0.0
            V_old = self.value_function.copy()
            # 行動価値関数更新
            q_new = np.zeros_like(self.q_function)
            for i, status in enumerate(self.environment.status):
                for j, action in enumerate(self.actions):
                    effect_action = self.effect_action(action)
                    reward = self.environment.give_reward(status, effect_action)
                    status_next = status + effect_action
                    p = 1           # 状態遷移は行動が決まれば決定的=1
                    q_new[i, j] += p * reward
                    if status_next >= 1 and status_next < self.q_function.shape[0]:
                        q_new[i, j] += p * (self.gamma * self.value_function[status_next - 1])
                    # 範囲の外
                    else:
                        q_new[i, j] += 0
                
                self.value_function[i] = np.max(q_new[i], axis=0)
                delta = max(np.abs(self.value_function[i] - V_old[i]), delta)
            
            self.q_function = q_new.copy()
            # 方策更新
            self.__decide_best_pi()
            # 履歴取得
            history_value_function.append(self.value_function.copy())
            
            if delta < threshold:
                break
        return history_value_function
    

if __name__ == "__main__":
    import environment
    environment = environment.Environment()
    agent = Agent(environment)
    agent.update_pi()