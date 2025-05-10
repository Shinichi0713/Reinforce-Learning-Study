

class Environment:
    def __init__(self):
        self.states = ['S', 'A', 'G']
        self.V = {'S': 0.0, 'A': 0.0, 'G': 0.0}

        # 1エピソードのオンライン学習
        self.trajectory = [('S', 'A', 0),  # (現状態, 次状態, 報酬)
                           ('A', 'G', 1)]  # ゴール到達時のみ報酬1
    
    # 状況に応じた報酬を与える
    def __give_reward(self, status, action):
        if status == 'A' and action == 1:
            return 1
        else:
            return 0

    # 次状態と、報酬を返す
    def get_action(self, status, action):
        reward = self.__give_reward(status, action)
        match status:
            case 'S':
                if action == 1: 
                    status_next = 'A'
                elif action == -1:
                    status_next = 'S'
                else:
                    status_next = 'S'
            case 'A':
                if action == 1:
                    status_next = 'G'
                elif action == 0:
                    status_next = 'A'
                else:
                    status_next = 'S'
            case _:
                status_next = 'S'
        return reward, status_next
        
