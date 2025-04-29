import numpy as np

# 迷路の設定
maze_size = (10, 10)
gamma = 0.8
reward_goal = 50
reward_wall = -1
reward_default = 0

# 迷路を初期化
maze = np.zeros(maze_size)
goal_position = (9, 9)  # ゴール位置
maze[goal_position] = reward_goal

# 状態価値関数を初期化
value_function = np.zeros(maze_size)

# 移動可能なアクション
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右, 下, 左, 上

# 報酬関数
def get_reward(position, next_position):
    if next_position[0] < 0 or next_position[0] >= maze_size[0] or next_position[1] < 0 or next_position[1] >= maze_size[1]:
        return reward_wall  # 壁にぶつかった場合の報酬
    elif next_position == goal_position:
        return reward_goal  # ゴールに到達した場合の報酬
    else:
        return reward_default  # 通常マス

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

# 全てのセルの状態価値関数を計算
for i in range(maze_size[0]):
    for j in range(maze_size[1]):
        value_function[i, j] = compute_value((i, j))

print("状態価値関数:")
print(value_function)
