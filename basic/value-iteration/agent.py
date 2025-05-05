import numpy as np


class Agent:
    def __init__(self, environment):
        self.actions = ["left", "right"]
        self.value_function = np.zeros(4)
        self.pi = np.array([[0.5, 0.5] for _ in range(4)])
        self.environment = environment

    def effect_action(self, action):
        if action == 'left':
            return -1
        else:
            return 1

    def evaluate_q_function(self, state, action, reward, next_state, gamma):
        reward = self.environment.give_reward(state, action)
        return reward + gamma * np.max(self.value_function[next_state])
    
if __name__ == "__main__":
    import environment
    env = environment.Environment()
    agent = Agent(env)
    index_action = agent.effect_action("left")
    print("Initial value function:", agent.value_function)
    print("Initial policy:", agent.pi)
    
