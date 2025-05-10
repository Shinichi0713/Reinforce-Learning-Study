# エージェントクラス
import numpy as np

class Agent:
    def __init__(self, env):
        self.status = None
        self.env = env
        
        self.pi = np.array([[1/3, 1/3, 1/3] for _ in range(len(self.env.states))])
        self.actions = np.array([-1, 0, 1])
        self.V = np.zeros(len(self.env.states))
        

    def calculate_td_error(self, state, action):
        # 次の状態へのtd-eerorを算出
        td_error = 0
        for i, prob_action in enumerate(self.pi):
            
            reward, state_next = self.env.get_action()

            td_error += prob_action * (reward + self.V[state_next] - self.V[state])

if __name__ == "__main__":
    import environment
    env = environment.Environment()
    agent_instance = Agent(env=env)
    agent_instance.select_action(state=None, action=None)
