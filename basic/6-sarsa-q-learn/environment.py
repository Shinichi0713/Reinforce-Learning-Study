import numpy as np


class Environment:
    def __init__(self):
        maze = [
            ['S', '.', '.', '#', '.'],
            ['.', '#', '.', '#', '.'],
            ['.', '#', '.', '.', '.'],
            ['.', '.', '#', '#', '.'],
            ['#', '.', '.', 'G', '.']
        ]
        self.maze = np.array(maze)
        self.position_start = self.__find('S')
        self.position_goal = self.__find('G')

    def __find(self, char):
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                if self.maze[r][c] == char:
                    return [r, c]
        return None

    def step(self, status, action):
        status_next = [status[0] + action[0], status[1] + action[1]]
        reward = self.give_reward(status_next)
        return reward, status_next

    def reset(self):
        self.position_start = self.__find('S')
        self.position_goal = self.__find('G')
        return self.position_start
    
    def give_reward(self, status_next):
        # ゴール
        if status_next == self.position_goal:
            return 2  # ゴールに到達
        elif 0 > status_next[0] or 0 > status_next[1] or status_next[0] >= self.maze.shape[0] or status_next[1] >= self.maze.shape[1]:
            return -2  # 壁に衝突
        elif self.maze[status_next[0], status_next[1]] == '#':
            return -1  # 壁に衝突
        return 0  # それ以外
    

if __name__ == "__main__":
    env = Environment()
    state = env.reset()
    done = False
    actions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    while not done:
        index_action = np.random.choice([0, 1, 2, 3])  # ランダムな行動を選択
        reward, state = env.step(state, actions[index_action])
        if reward == 2:
            done = True
    print("Episode finished")