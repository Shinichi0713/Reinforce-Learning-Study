
import numpy as np
import math, time


class Agent():
    def __init__(self, width_screen, height_screen):
        self.q_table = np.zeros((16, 4))
        self.num_actions = 4
        self.num_width_split = 10
        self.num_height_split = 10
        self.width_split = width_screen // self.num_width_split
        self.height_split = height_screen // self.num_height_split

    # 状態を取得
    # マリオとクリボーの座標
    # マリオとクリボーの速度
    def interpret_status(self, position_mario, position_enemy, velocity_mario, velocity_enemy):
        state = []
        state.append(position_mario[0] // self.width_split)
        state.append(position_mario[1] // self.height_split)
        state.append(position_enemy[0] // self.width_split)
        state.append(position_enemy[1] // self.height_split)
        return state

    def update_q_table(self, state, action, reward, state_next):
        alpha = 0.2
        gamma = 0.99
        q_value_max = max(self.q_table[state_next])
        q_value_current = self.q_table[state, action]
        # qテーブルの更新＝Q学習で更新
        self.q_table[state, action] = q_value_current \
            + alpha * (reward + gamma * q_value_max - q_value_current)
        self.q_table_update[state] = [math.floor(time.time()), self.q_table_update[state][1] + 1]