import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Tuple, Dict, Optional

class CooperativeGridEnv:
    """
    2エージェント協調収集グリッドワールド環境
    - グリッドサイズ: 3x3
    - 資源は中央 (1,1) に固定
    - エージェントは上下左右＋待機の5行動
    - 協調収集で高報酬
    """
    def __init__(self, grid_size: int = 3):
        self.grid_size = grid_size
        self.resource_pos = (1, 1)  # 資源の位置（中央）
        self.n_agents = 2
        self.action_space = [0, 1, 2, 3, 4]  # 上, 下, 左, 右, 待機
        self.action_meanings = ["上", "下", "左", "右", "待機"]

        # 初期位置（左上と右下）
        self.init_positions = [(0, 0), (grid_size-1, grid_size-1)]

    def reset(self) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """
        環境をリセットし、初期状態と観測を返す
        """
        self.agent_positions = self.init_positions.copy()
        observations = self._get_observations()
        return self.agent_positions, observations

    def _get_observations(self) -> List[np.ndarray]:
        """
        各エージェントの観測を返す
        - 観測: [自分のx, 自分のy, 相手のx, 相手のy]
        """
        obs = []
        for i in range(self.n_agents):
            my_x, my_y = self.agent_positions[i]
            other_x, other_y = self.agent_positions[1 - i]  # 相手
            obs.append(np.array([my_x, my_y, other_x, other_y], dtype=np.float32))
        return obs

    def step(self, actions: List[int]) -> Tuple[List[Tuple[int, int]], List[np.ndarray], List[float], bool, Dict]:
        """
        2エージェントが行動を取り、次の状態・報酬などを返す
        """
        assert len(actions) == self.n_agents

        # 1. 移動の計算（壁チェック付き）
        new_positions = []
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            if action == 0:   # 上
                y = max(0, y - 1)
            elif action == 1: # 下
                y = min(self.grid_size - 1, y + 1)
            elif action == 2: # 左
                x = max(0, x - 1)
            elif action == 3: # 右
                x = min(self.grid_size - 1, x + 1)
            elif action == 4: # 待機
                pass
            new_positions.append((x, y))

        # 2. 衝突チェック（同じマスに行こうとしたら元の位置に留まる）
        if new_positions[0] == new_positions[1]:
            # 衝突ペナルティ
            rewards = [-0.1, -0.1]
            # 位置は更新しない
            self.agent_positions = self.agent_positions
        else:
            self.agent_positions = new_positions
            rewards = [0.0, 0.0]

        # 3. 資源収集の報酬計算
        resource_x, resource_y = self.resource_pos
        # 各エージェントが資源に隣接しているかチェック
        adjacent_to_resource = []
        for (x, y), action in zip(self.agent_positions, actions):
            # 資源マスに隣接しているか（上下左右）
            is_adjacent = (
                (abs(x - resource_x) + abs(y - resource_y) == 1) and
                (action != 4)  # 待機でない
            )
            adjacent_to_resource.append(is_adjacent)

        # 単独収集 or 協調収集
        if sum(adjacent_to_resource) == 1:
            # 単独収集: +1
            for i in range(self.n_agents):
                if adjacent_to_resource[i]:
                    rewards[i] += 1.0
        elif sum(adjacent_to_resource) == 2:
            # 協調収集: 各エージェントに +2
            for i in range(self.n_agents):
                rewards[i] += 2.0

        # 4. 観測と終了判定（ここでは無限エピソード）
        observations = self._get_observations()
        done = False  # 簡易のため終了条件は設けない
        info = {}

        return self.agent_positions, observations, rewards, done, info

    def render(self, ax):
        """
        現在の状態を matplotlib Axes に描画
        """
        ax.clear()
        grid_size = self.grid_size
        resource_pos = self.resource_pos

        # グリッド線
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')

        # 資源を描画
        rx, ry = resource_pos
        ax.add_patch(plt.Rectangle((rx - 0.5, ry - 0.5), 1, 1, color='gold', alpha=0.7))

        # エージェントを描画
        colors = ['red', 'blue']
        labels = ['Agent 1', 'Agent 2']
        for i, (x, y) in enumerate(self.agent_positions):
            ax.add_patch(plt.Circle((x, y), 0.3, color=colors[i], alpha=0.8))
            ax.text(x, y, labels[i][-1], ha='center', va='center', fontweight='bold')

    def save_gif(self, trainer=None, filename="cooperative_grid.gif", max_steps: int = 20):
        """
        1エピソードを実行し、GIFとして保存する
        - trainer が None の場合はランダム行動
        - trainer がある場合はモデルから行動を選択（ここでは未実装）
        """
        frames = []
        obs_list = self.reset()[1]  # 観測のみ取得
        done = False
        step_count = 0

        fig, ax = plt.subplots(figsize=(5, 5))

        while not done and step_count < max_steps:
            # 現在の状態を描画してフレームに追加
            self.render(ax)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))

            # 行動選択（trainer がなければランダム）
            if trainer is None:
                actions = [np.random.randint(0, 5) for _ in range(self.n_agents)]
            else:
                # ここに QMIX や MAVEN などのモデルから行動を選択するコードを追加
                # 例: actions = trainer.select_actions(obs_list)
                actions = [np.random.randint(0, 5) for _ in range(self.n_agents)]  # 仮実装

            # 環境を1ステップ進める
            _, obs_list, _, done, _ = self.step(actions)
            step_count += 1

        # GIFとして保存
        if frames:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=300,  # 各フレームの表示時間（ms）
                loop=0        # ループ回数（0=無限）
            )
            plt.close(fig)
            print(f"✅ GIF saved as {filename}")
        else:
            plt.close(fig)
            print("❌ No frames to save")


# --- 実行例 ---
if __name__ == "__main__":
    env = CooperativeGridEnv(grid_size=3)

    # GIFを保存（ランダム行動）
    env.save_gif(trainer=None, filename="cooperative_grid_random.gif", max_steps=15)

    # 学習済みモデルがある場合は trainer を渡す（ここでは未実装）
    # env.save_gif(trainer=my_trainer, filename="cooperative_grid_trained.gif", max_steps=15)