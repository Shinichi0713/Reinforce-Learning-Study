
import numpy as np


class Environment:
    def __init__(self):
        self.states = ['S', 'A', 'G']
        self.V = {'S': 0.0, 'A': 0.0, 'G': 0.0}

        # 1エピソードのオンライン学習
        self.trajectory = [('S', 'A', 0),  # (現状態, 次状態, 報酬)
                           ('A', 'G', 1)]  # ゴール到達時のみ報酬1
        
    def give_reward(self, status, action):
        if status == 'A' and action == 1:
            return 1
        else:
            return 0
