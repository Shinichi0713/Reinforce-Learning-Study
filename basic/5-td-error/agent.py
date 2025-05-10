# エージェントクラス
import numpy as np

class Agent:
    def __init__(self, env):
        self.status = None
        self.env = env
        
        self.pi = np.array([[1/3, 1/3, 1/3] for _ in range(len(self.env.states))])
        self.actions = np.array([-1, 0, 1])
        self.V = np.zeros(len(self.env.states))
        

    # td誤差の計算
    def calculate_td_error(self, state):
        # 次の状態へのtd-eerorを算出
        td_error = 0
        for i, prob_action in enumerate(self.pi[self.env.return_index_status(state)]):
            action = self.actions[i]
            reward, state_next = self.env.get_action(state, action)
            index_state = self.env.return_index_status(state)
            index_state_next = self.env.return_index_status(state_next)
            td_error += prob_action * (reward + self.V[index_state_next] - self.V[index_state])
        return td_error

if __name__ == "__main__":
    import environment
    env = environment.Environment()
    agent_instance = Agent(env=env)
    print(agent_instance.calculate_td_error('A'))
