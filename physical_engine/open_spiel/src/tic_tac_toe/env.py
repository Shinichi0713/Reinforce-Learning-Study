import numpy as np
import pyspiel
import pygame
import os
import imageio

os.makedirs("/tmp/frames", exist_ok=True)

class TicTacToeEnv:
    def __init__(self):
        self.game = pyspiel.load_game("tic_tac_toe")
        self.state = self.game.new_initial_state()
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 の盤面 (0:空, 1:プレイヤー0, 2:プレイヤー1)

    def step(self, action):
        self.state.apply_action(action)
        # 行動履歴から盤面を更新
        self._update_board_from_history()
        return self.state

    def reset(self):
        self.state = self.game.new_initial_state()
        self.board = np.zeros((3, 3), dtype=int)
        return self.state

    def is_terminal(self):
        return self.state.is_terminal()

    def legal_actions(self):
        return self.state.legal_actions()

    def _update_board_from_history(self):
        # 行動履歴から盤面を再構築
        self.board = np.zeros((3, 3), dtype=int)
        history = self.state.history()
        for i, action in enumerate(history):
            player = i % 2  # 0: プレイヤー0, 1: プレイヤー1
            row = action // 3
            col = action % 3
            self.board[row, col] = player + 1  # 1 or 2

    def get_board_representation(self):
        return self.board.copy()

class TicTacToeRenderer:
    def __init__(self, cell_size=100, margin=10):
        self.cell_size = cell_size
        self.margin = margin
        self.width = 3 * cell_size + 4 * margin
        self.height = 3 * cell_size + 4 * margin
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))

    def draw_board(self, board):
        self.screen.fill((255, 255, 255))
        for i in range(3):
            for j in range(3):
                x = j * (self.cell_size + self.margin) + self.margin
                y = i * (self.cell_size + self.margin) + self.margin
                pygame.draw.rect(self.screen, (200, 200, 200),
                                 (x, y, self.cell_size, self.cell_size), 2)
                if board[i, j] == 1:
                    # X (プレイヤー0)
                    pygame.draw.line(self.screen, (255, 0, 0),
                                     (x + 10, y + 10),
                                     (x + self.cell_size - 10, y + self.cell_size - 10), 3)
                    pygame.draw.line(self.screen, (255, 0, 0),
                                     (x + self.cell_size - 10, y + 10),
                                     (x + 10, y + self.cell_size - 10), 3)
                elif board[i, j] == 2:
                    # O (プレイヤー1)
                    pygame.draw.circle(self.screen, (0, 0, 255),
                                      (x + self.cell_size // 2, y + self.cell_size // 2),
                                      self.cell_size // 2 - 10, 3)
        return self.screen

def main():
    env = TicTacToeEnv()
    renderer = TicTacToeRenderer()

    frame_files = []
    frame_count = 0

    # 初期状態 (空の盤面)
    board = env.get_board_representation()
    screen = renderer.draw_board(board)
    pygame.image.save(screen, f"/tmp/frames/frame_{frame_count:04d}.png")
    frame_files.append(f"/tmp/frames/frame_{frame_count:04d}.png")
    frame_count += 1

    # ランダムプレイ
    while not env.is_terminal():
        action = np.random.choice(env.legal_actions())
        env.step(action)

        board = env.get_board_representation()
        screen = renderer.draw_board(board)
        pygame.image.save(screen, f"/tmp/frames/frame_{frame_count:04d}.png")
        frame_files.append(f"/tmp/frames/frame_{frame_count:04d}.png")
        frame_count += 1

    # 終端状態の盤面をもう一度描画（最終フレーム）
    board = env.get_board_representation()
    screen = renderer.draw_board(board)
    pygame.image.save(screen, f"/tmp/frames/frame_{frame_count:04d}.png")
    frame_files.append(f"/tmp/frames/frame_{frame_count:04d}.png")
    frame_count += 1

    pygame.quit()

    # 動画作成
    with imageio.get_writer("/tmp/tic_tac_toe.mp4", fps=2) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("動画を保存しました: /tmp/tic_tac_toe.mp4")

if __name__ == "__main__":
    main()