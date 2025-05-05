import numpy as np


class Environment:
    def __init__(self):
        self.status = np.array([1, 2, 3, 4])

    def give_reward(self, status, action):
        status_next = status + action
        if status == 3 and status_next == 4:
            return 10
        else:
            return 0

    def is_terminal(self, status):
        return status == 4