import numpy as np
import gymnasium as gym
from gymnasium import spaces

BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

class GoEnv(gym.Env):
    """
    RaylibのGoGameクラスと完全同一のルールを持つGymnasium環境
    """
    def __init__(self):
        super(GoEnv, self).__init__()
        
        # 1. 行動空間: 19x19 = 361 位置 (パス行動を追加する場合は 362)
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)
        
        # 2. 観測空間: (3, 19, 19) チャネル構成
        # Channel 0: 自分の石の位置 (1 or 0)
        # Channel 1: 相手の石の位置 (1 or 0)
        # Channel 2: 手番 (自分が黒なら全面1, 白なら全面0)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        
        self.dr = [-1, 1, 0, 0]
        self.dc = [0, 0, -1, 1]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_turn = BLACK
        self.black_captures = 0
        self.white_captures = 0
        self.step_count = 0
        return self._get_obs(), {}

    def _is_valid(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _get_group_and_liberties(self, r, c):
        color = self.board[r, c]
        if color == EMPTY:
            return set(), set()

        group = set([(r, c)])
        liberties = set()
        stack = [(r, c)]

        while stack:
            curr_r, curr_c = stack.pop()
            for i in range(4):
                nr, nc = curr_r + self.dr[i], curr_c + self.dc[i]
                if not self._is_valid(nr, nc):
                    continue

                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    stack.append((nr, nc))

        return group, liberties

    def step(self, action):
        r = action // BOARD_SIZE
        c = action % BOARD_SIZE
        
        # 非合法手（既に石がある・範囲外）のチェック
        if not self._is_valid(r, c) or self.board[r, c] != EMPTY:
            # 負の報酬を与えてエピソード終了（または手番スキップ）
            return self._get_obs(), -10.0, True, False, {"error": "Invalid move"}

        opponent = WHITE if self.current_turn == BLACK else BLACK
        self.board[r, c] = self.current_turn

        # 1. 相手の石の獲得チェック
        captured = 0
        for i in range(4):
            nr, nc = r + self.dr[i], c + self.dc[i]
            if self._is_valid(nr, nc) and self.board[nr, nc] == opponent:
                group, liberties = self._get_group_and_liberties(nr, nc)
                if len(liberties) == 0:
                    for pr, pc in group:
                        self.board[pr, pc] = EMPTY
                        captured += 1

        if self.current_turn == BLACK:
            self.black_captures += captured
        else:
            self.white_captures += captured

        # 2. 自殺手の判定
        my_group, my_liberties = self._get_group_and_liberties(r, c)
        if len(my_liberties) == 0 and captured == 0:
            self.board[r, c] = EMPTY
            return self._get_obs(), -10.0, True, False, {"error": "Suicide move"}

        # 状態更新
        self.step_count += 1
        reward = float(captured) # アゲハ獲得による即時報酬（必要に応じて変更）
        
        # 手番交代
        self.current_turn = opponent
        
        # 最大ステップ（例: 300手）で終了
        done = self.step_count >= 300
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        """ニューラルネットワーク入力用の状態生成"""
        my_color = self.current_turn
        opp_color = WHITE if my_color == BLACK else BLACK

        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board == my_color).astype(np.float32)
        obs[1] = (self.board == opp_color).astype(np.float32)
        obs[2] = 1.0 if my_color == BLACK else 0.0

        return obs

    def get_legal_actions(self):
        """合法手マスクの取得 (1: 合法, 0: 非合法)"""
        mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.bool_)
        for a in range(BOARD_SIZE * BOARD_SIZE):
            r, c = a // BOARD_SIZE, a % BOARD_SIZE
            if self.board[r, c] == EMPTY:
                # 簡略化したチェック（必要に応じて完全検証）
                mask[a] = True
        return mask