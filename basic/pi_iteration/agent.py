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

    # その段階での行動価値関数を評価する
    def evaluate_q_function(self):
        for i, status in enumerate(self.environment.status):
            for j, action in enumerate(self.actions):
                effect_action = self.effect_action(action)
                reward = self.environment.give_reward(status, effect_action)
                pi = self.pi[i, j]
                status_next = status + effect_action
                p = 1           # 状態遷移は行動が決まれば決定的=1
                self.value_function[i] += pi * p * reward
                if status_next >= 0 and status_next < self.value_function.shape[0]:
                    self.value_function[i] += pi * p * (self.gamma * self.value_function[status_next])
                # 範囲の外
                else:
                    self.value_function[i] += 0
        return self.value_function
    
if __name__ == "__main__":
    import environment
    env = environment.Environment()
    agent = Agent(env)
    index_action = agent.effect_action("left")
    q_function = agent.evaluate_q_function()
    print("Q function:", q_function)
    
