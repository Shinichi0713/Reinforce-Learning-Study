import numpy as np
import random

# 迷路データ
maze = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '#', '#', '.'],
    ['#', '.', '.', 'G', '.']
]

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_TO_DELTA = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.n_row = len(maze)
        self.n_col = len(maze[0])
        self.start = self._find('S')
        self.goal = self._find('G')
        self.reset()

    def _find(self, char):
        for r in range(self.n_row):
            for c in range(self.n_col):
                if self.maze[r][c] == char:
                    return (r, c)
        return None

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        # 壁や範囲外は移動しない
        if 0 <= nr < self.n_row and 0 <= nc < self.n_col and self.maze[nr][nc] != '#':
            self.pos = (nr, nc)
        # 報酬設定
        if self.pos == self.goal:
            return self.pos, 10, True  # ゴール
        else:
            return self.pos, -1, False  # 移動

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.n_row and 0 <= c < self.n_col and self.maze[r][c] != '#'
