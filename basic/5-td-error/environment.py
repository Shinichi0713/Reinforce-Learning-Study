# 環境クラス
class Environment:
    def __init__(self):
        self.states = ['S', 'A', 'G']
        self.V = {'S': 0.0, 'A': 0.0, 'G': 0.0}

        # 1エピソードのオンライン学習
        self.trajectory = [('S', 'A', 0),  # (現状態, 次状態, 報酬)
                           ('A', 'G', 1)]  # ゴール到達時のみ報酬1
    
    def reset(self):
        self.V = {'S': 0.0, 'A': 0.0, 'G': 0.0}
        self.trajectory = [('S', 'A', 0),  # (現状態, 次状態, 報酬)
                           ('A', 'G', 1)]
        return self.states[0]

    # 状況に応じた報酬を与える
    def __give_reward(self, state, action):
        if state == 'A' and action == 1:
            return 1
        else:
            return 0

    # 次状態と、報酬を返す
    def get_action(self, state, action):
        reward = self.__give_reward(state, action)
        match state:
            case 'S':
                if action == 1: 
                    state_next = 'A'
                elif action == -1:
                    state_next = 'S'
                else:
                    state_next = 'S'
            case 'A':
                if action == 1:
                    state_next = 'G'
                elif action == 0:
                    state_next = 'A'
                else:
                    state_next = 'S'
            case _:
                state_next = 'S'
        return reward, state_next
    
    # 状態のインデクスを返す
    def return_index_status(self, state):
        return self.states.index(state) if state in self.states else -1