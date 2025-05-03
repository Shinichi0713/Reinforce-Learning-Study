# 環境(迷路)
import numpy as np
import os


reward_goal = 10
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
    def give_reward(self, agent_position, move):
        status, status_next = self.__give_status(agent_position, move)  # 修正: list()を削除し、正しいリストを作成
        if status_next == '1':
            return reward_wall  # 壁にぶつかった場合の報酬
        elif status == '0' and status_next == 'E':
            return reward_goal  # ゴールに到達した場合の報酬
        else:
            return reward_default
    
    # エージェントの位置と、移動方向に応じてstatusを返す
    def __give_status(self, coordinate, move):

        status = self.maze[coordinate[0], coordinate[1]]
        next_coordinate = (coordinate[0] + move[0], coordinate[1] + move[1])
        status_next = self.maze[next_coordinate] if 0 <= next_coordinate[0] < self.maze.shape[0] and 0 <= next_coordinate[1] < self.maze.shape[1] else '1'
        return status, status_next

    def show_start_position(self):
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 'S':
                    return [i, j]


if __name__ == "__main__":
    env = Environment()
    reward = env.give_reward([0, 2], [0,1])
    print("Reward:", reward)