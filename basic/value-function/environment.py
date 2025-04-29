# 環境
import numpy as np
import os


reward_goal = 50
reward_wall = -1
reward_default = 0


class Environment:
    # 迷路を初期化
    def __init__(self):
        
        dir_current = os.path.dirname(os.path.abspath(__file__))
        with open(f'{dir_current}/maze.txt', 'r') as f:
            maze_read = [s.rstrip().split(' ') for s in f.readlines()]
        self.maze = np.array(maze_read)

    # エージェントの状態に応じて報酬を返す
    def give_reward(self, coordinate, move):
        status, status_next = self.__give_status(coordinate, move)
        if status_next == '1':
            return reward_wall  # 壁にぶつかった場合の報酬
        elif status == '0' and status_next == 'E':
            return reward_goal  # ゴールに到達した場合の報酬
        else:
            return reward_default
    
    # エージェントの位置と、移動方向に応じてstatusを返す
    def __give_status(self, coordinate, move):

        status = self.maze[coordinate]
        next_coordinate = (coordinate[0] + move[0], coordinate[1] + move[1])
        status_next = self.maze[next_coordinate] if 0 <= next_coordinate[0] < maze_size[0] and 0 <= next_coordinate[1] < maze_size[1] else '1'
        return status, status_next

    def show_start_position(self):
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 'S':
                    return [i, j]


# 移動可能なアクション
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右, 下, 左, 上


# 再帰的な状態価値関数
def compute_value(position, depth=10):
    if depth == 0 or position == goal_position:
        return 0

    value = 0
    for action in actions:
        next_position = (position[0] + action[0], position[1] + action[1])
        reward = get_reward(position, next_position)
        next_value = compute_value(next_position, depth - 1)
        value += (1 / len(actions)) * (reward + gamma * next_value)  # 等確率で移動

    return value


if __name__ == "__main__":
    env = Environment()
    start_position = env.show_start_position()
    print("Start Position:", start_position)