import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MazeEnv:
    """
    5x5迷路環境クラス（動画保存機能付き）
    """
    def __init__(self, maze_file="maze.txt", rows=5, cols=5):
        if os.path.exists(maze_file):
            self.maze = self._load_maze(maze_file)
            self.start, self.goal = self._find_start_goal()
        else:
            # 自動生成メソッドを呼び出す
            self.generate_random_maze(rows=rows, cols=cols)
            
        self.state = self.start
        self.done = False
        self.history = []

    def _load_maze(self, file_path):
        maze = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ' ' in line:
                    row = line.split()
                else:
                    row = list(line)
                maze.append(row)
        return maze

    def _find_start_goal(self):
        start = None
        goal = None
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == 'S':
                    start = (i, j)
                elif cell == 'G':
                    goal = (i, j)
        return start, goal

    def _is_valid_move(self, pos):
        x, y = pos
        if x < 0 or y < 0:
            return False
        if x >= len(self.maze) or y >= len(self.maze[0]):
            return False
        if self.maze[x][y] == 'W':
            return False
        return True

    def reset(self):
        self.state = self.start
        self.done = False
        self.history = [self.state]  # 履歴もリセット
        return self.state

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already done. Call reset() first.")

        x, y = self.state
        if action == 0:   # 上
            next_state = (x - 1, y)
        elif action == 1: # 下
            next_state = (x + 1, y)
        elif action == 2: # 左
            next_state = (x, y - 1)
        elif action == 3: # 右
            next_state = (x, y + 1)
        else:
            raise ValueError("Invalid action")

        if not self._is_valid_move(next_state):
            next_state = self.state
            reward = -1
        else:
            reward = 0

        if self.maze[next_state[0]][next_state[1]] == 'G':
            reward = 10
            self.done = True
        else:
            self.done = False

        self.state = next_state
        self.history.append(self.state)  # 履歴に追加
        return self.state, reward, self.done

    def render(self):
        maze_copy = [row[:] for row in self.maze]
        x, y = self.state
        if not self.done:
            maze_copy[x][y] = 'A'
        for row in maze_copy:
            print(' '.join(row))
        print()

    def get_image_observation(self):
        """
        迷路を画像（NumPy配列）として返す
        チャネル0: 壁 (W) = 1, それ以外 = 0
        チャネル1: スタート (S) = 1, それ以外 = 0
        チャネル2: ゴール (G) = 1, それ以外 = 0
        チャネル3: エージェント位置 (A) = 1, それ以外 = 0
        """
        rows = len(self.maze)
        cols = len(self.maze[0])
        obs = np.zeros((4, rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                if cell == 'W':
                    obs[0, i, j] = 1.0
                elif cell == 'S':
                    obs[1, i, j] = 1.0
                elif cell == 'G':
                    obs[2, i, j] = 1.0
                elif cell == '.':
                    pass  # 何も立てない
        # エージェント位置
        x, y = self.state
        obs[3, x, y] = 1.0
        return obs

    def save_video(self, output_path="maze_animation.mp4", fps=2):
        """
        エージェントの動きを動画として保存
        """
        # 迷路のサイズ
        rows = len(self.maze)
        cols = len(self.maze[0])

        # カラーマップの定義
        cell_colors = {
            'S': 'lightblue',  # スタート
            'G': 'lightgreen', # ゴール
            'W': 'black',      # 壁
            '.': 'white',      # 通路
            'A': 'red'         # エージェント（描画時に上書き）
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        # 背景（迷路）を描画
        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                color = cell_colors.get(cell, 'white')
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                # ラベル（S, G, W, .）を表示
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        # エージェントの位置を示すマーカー
        agent_marker, = ax.plot([], [], 'o', markersize=20, color='red')

        def init():
            agent_marker.set_data([], [])
            return agent_marker,

        def update(frame):
            if frame >= len(self.history):
                return agent_marker,
            x, y = self.history[frame]
            # 座標変換（matplotlibはy軸が下向きなので反転）
            plot_y = rows - x - 1
            agent_marker.set_data([y], [plot_y])
            return agent_marker,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                            init_func=init, blit=True, interval=1000/fps)

        # MP4として保存（ffmpegが必要）
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Animation saved to {output_path}")


    def generate_random_maze(self, rows=5, cols=5, wall_prob=0.3):
        """
        条件を満たす迷路を自動生成し、self.mazeにセットする
        条件:
        1. S (スタート) と G (ゴール) が必ず存在する
        2. S から G へ必ず到達できる
        3. S と G の間には適度に障害物（W）が存在する
        """
        import random
        
        while True:
            # 1. すべてを通路('.')で初期化
            maze = [['.' for _ in range(cols)] for _ in range(rows)]
            
            # 2. スタートとゴールをランダムな位置に配置（重複しないように）
            s_r, s_c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            while True:
                g_r, g_c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                if (g_r, g_c) != (s_r, s_c):
                    break
            
            maze[s_r][s_c] = 'S'
            maze[g_r][g_c] = 'G'
            
            # 3. 確率に基づいて壁('W')を配置
            for r in range(rows):
                for c in range(cols):
                    if maze[r][c] not in ['S', 'G']:
                        if random.random() < wall_prob:
                            maze[r][c] = 'W'
            
            # 4. 幅優先探索（BFS）でSからGへの経路が存在するか（到達可能か）チェック
            queue = [(s_r, s_c)]
            visited = set([(s_r, s_c)])
            reachable = False
            
            while queue:
                curr_r, curr_c = queue.pop(0)
                
                if (curr_r, curr_c) == (g_r, g_c):
                    reachable = True
                    break
                    
                # 上下左右の移動
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if maze[nr][nc] != 'W' and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            
            # 5. 到達可能、かつ迷路内に少なくとも1つは壁が存在することを確認
            # （5x5だとたまに壁が1つも生成されないことがあるため）
            has_wall = any('W' in row for row in maze)
            
            if reachable and has_wall:
                self.maze = maze
                self.start = (s_r, s_c)
                self.goal = (g_r, g_c)
                break