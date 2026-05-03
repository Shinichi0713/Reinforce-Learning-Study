import numpy as np
import pyspiel
import pygame
import os
import imageio

# Google Colab で pygame をヘッドレスで動かすための設定
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 画面表示を無効化（ヘッドレス）
pygame.init()

os.makedirs("/tmp/go_frames", exist_ok=True)

class GoEnv:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.game = pyspiel.load_game(f"go(board_size={board_size})")
        self.state = self.game.new_initial_state()
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0:空, 1:黒, 2:白

    def step(self, action):
        self.state.apply_action(action)
        self._update_board_from_state()
        return self.state

    def reset(self):
        self.state = self.game.new_initial_state()
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        return self.state

    def is_terminal(self):
        return self.state.is_terminal()

    def legal_actions(self):
        return self.state.legal_actions()

    def _update_board_from_state(self):
        # OpenSpiel の状態から盤面を再構築
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # OpenSpiel の Go 状態は observation_tensor などで盤面を取得可能
        # ここでは簡易的に history から再構築（実用上は observation_tensor を使う方が正確）
        history = self.state.history()
        for i, action in enumerate(history):
            player = i % 2  # 0: 黒, 1: 白
            if action == self.board_size * self.board_size:  # パス
                continue
            row = action // self.board_size
            col = action % self.board_size
            self.board[row, col] = player + 1  # 1: 黒, 2: 白

    def get_board_representation(self):
        return self.board.copy()

class GoRenderer:
    def __init__(self, board_size=9, cell_size=40, margin=20):
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.width = board_size * cell_size + 2 * margin
        self.height = board_size * cell_size + 2 * margin
        self.screen = pygame.Surface((self.width, self.height))

    def draw_board(self, board):
        self.screen.fill((220, 180, 100))  # 木目風の背景
        # グリッド線
        for i in range(self.board_size):
            # 横線
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.margin, self.margin + i * self.cell_size),
                (self.width - self.margin, self.margin + i * self.cell_size),
                1
            )
            # 縦線
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.margin + i * self.cell_size, self.margin),
                (self.margin + i * self.cell_size, self.height - self.margin),
                1
            )

        # 星（9路盤の場合）
        if self.board_size == 9:
            stars = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
            for row, col in stars:
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 3)

        # 石の描画
        stone_radius = self.cell_size // 2 - 2
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 1:  # 黒石
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), stone_radius)
                elif board[i, j] == 2:  # 白石
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), stone_radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), stone_radius, 1)

        return self.screen

def record_go_game(output_path="/tmp/go_demo.mp4", fps=1):
    """
    囲碁のランダムプレイを録画（PNG→MP4方式）
    """
    env = GoEnv(board_size=9)
    renderer = GoRenderer(board_size=9, cell_size=40, margin=20)

    # フレーム保存用リスト（PNGファイル名）
    frame_files = []
    frame_count = 0

    # 初期盤面を保存
    board = env.get_board_representation()
    screen = renderer.draw_board(board)
    pygame.image.save(screen, f"/tmp/go_frames/frame_{frame_count:04d}.png")
    frame_files.append(f"/tmp/go_frames/frame_{frame_count:04d}.png")
    frame_count += 1

    # ランダムプレイ
    while not env.is_terminal():
        legal_actions = list(env.legal_actions())
        if legal_actions:
            action = np.random.choice(legal_actions)
            env.step(action)

            board = env.get_board_representation()
            screen = renderer.draw_board(board)
            pygame.image.save(screen, f"/tmp/go_frames/frame_{frame_count:04d}.png")
            frame_files.append(f"/tmp/go_frames/frame_{frame_count:04d}.png")
            frame_count += 1
        else:
            break

    # 終端状態をもう一度保存（最終フレーム）
    screen = renderer.draw_board(board)
    pygame.image.save(screen, f"/tmp/go_frames/frame_{frame_count:04d}.png")
    frame_files.append(f"/tmp/go_frames/frame_{frame_count:04d}.png")
    frame_count += 1

    # PNGからMP4に変換
    with imageio.get_writer(output_path, fps=fps) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"囲碁動画を保存しました: {output_path}")

    # Colab 上で動画を表示
    from IPython.display import display, HTML
    display(HTML(f"""
    <video width="400" controls>
      <source src="{output_path}" type="video/mp4">
    </video>
    """))

