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
        self.reset()

    def __find(self, char):
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                if self.maze[r][c] == char:
                    return [r, c]
        return None

    def step(self, status, action):
        status_next = status + np.array(action)
        reward, status_next, done = self.give_reward(status_next)
        if done:
            self.status = self.reset()
        return reward, status_next, done

    def reset(self):
        self.position_start = np.array(self.__find('S'))
        self.position_goal = np.array(self.__find('G'))
        return self.position_start
    
    def give_reward(self, status_next):
        # ゴール
        if status_next.all() == self.position_goal.all():
            return 2, status_next, True  # ゴールに到達
        elif 0 > status_next[0] or 0 > status_next[1] or status_next[0] >= self.maze.shape[0] or status_next[1] >= self.maze.shape[1]:
            status_next = np.clip(status_next, 0, self.maze.shape[0] - 1)
            return -2, status_next, False  # 壁に衝突
        elif self.maze[status_next[0], status_next[1]] == '#':
            return -1, status_next, False  # 壁に衝突
        return 0, status_next, False  # それ以外

    def check_over(self, state):
        # エピソードが終了したかどうかをチェック
        return state.all() == self.position_goal.all()

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