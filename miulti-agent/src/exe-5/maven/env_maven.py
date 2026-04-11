import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os
from typing import List, Tuple, Dict, Optional

class CooperativeGridEnv:
    def __init__(self, grid_size: int = 3, max_steps: int = 20):
        self.grid_size = grid_size
        self.resource_pos = (1, 1)
        self.n_agents = 2
        self.action_space = [0, 1, 2, 3, 4]
        self.action_meanings = ["up", "down", "left", "right", "stay"]  # 英語に変更
        self.init_positions = [(0, 0), (grid_size-1, grid_size-1)]
        self.max_steps = max_steps
        self.step_count = 0
        self.last_actions = [4, 4]  # 初期は "stay"

    def reset(self):
        self.agent_positions = self.init_positions.copy()
        self.step_count = 0
        self.last_actions = [4, 4]  # 初期は "stay"
        observations = self._get_observations()
        return self.agent_positions, observations

    def step(self, actions: List[int]):
        assert len(actions) == self.n_agents
        self.step_count += 1
        self.last_actions = actions  # 行動を保存

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

        # 2. 衝突チェック
        if new_positions[0] == new_positions[1]:
            rewards = [-0.1, -0.1]
            self.agent_positions = self.agent_positions
        else:
            self.agent_positions = new_positions
            rewards = [0.0, 0.0]

        # 3. 資源収集の報酬計算（スケールアップ）
        resource_x, resource_y = self.resource_pos
        adjacent_to_resource = []
        for (x, y), action in zip(self.agent_positions, actions):
            is_adjacent = (
                (abs(x - resource_x) + abs(y - resource_y) == 1) and
                (action != 4)
            )
            adjacent_to_resource.append(is_adjacent)

        if sum(adjacent_to_resource) == 1:
            # 単独収集: +10
            for i in range(self.n_agents):
                if adjacent_to_resource[i]:
                    rewards[i] += 10.0
        elif sum(adjacent_to_resource) == 2:
            # 協調収集: 各エージェントに +20
            for i in range(self.n_agents):
                rewards[i] += 20.0

        # 4. 終了判定（max_steps で終了）
        observations = self._get_observations()
        done = (self.step_count >= self.max_steps)
        info = {}

        return self.agent_positions, observations, rewards, done, info

    def _get_observations(self):
        obs = []
        for i in range(self.n_agents):
            my_x, my_y = self.agent_positions[i]
            other_x, other_y = self.agent_positions[1 - i]
            obs.append(np.array([my_x, my_y, other_x, other_y], dtype=np.float32))
        return obs

    def render(self, ax):
        """
        現在のグリッド状態を matplotlib の ax に描画
        - エージェントの位置と行動を表示（Colab 対応）
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

        # 資源マス
        rx, ry = resource_pos
        ax.add_patch(plt.Rectangle((rx - 0.5, ry - 0.5), 1, 1, color='gold', alpha=0.7))

        # エージェントと行動を描画
        colors = ['red', 'blue']
        labels = ['Agent 1', 'Agent 2']
        for i, (x, y) in enumerate(self.agent_positions):
            # エージェントの円
            ax.add_patch(plt.Circle((x, y), 0.3, color=colors[i], alpha=0.8))
            # エージェント番号（円の中心）
            ax.text(x, y, labels[i][-1], ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            # 行動をテキストで表示（円の上に少し離して）
            action_text = self.action_meanings[self.last_actions[i]]
            ax.text(x, y + 0.5, action_text, ha='center', va='bottom', fontsize=10, color=colors[i], fontweight='bold')

        ax.set_title("Cooperative Grid World")


class MavenStyleMultiAgent:
    """
    MAVEN風の多モード探索を行う2エージェント用エージェント（簡易版）
    - 各エージェントが独立に ε-greedy
    - モード切り替えで探索の多様性を模倣
    """
    def __init__(self, n_agents: int, n_actions: int, state_dim: int,
                 epsilon1: float = 0.1, epsilon2: float = 0.5):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

        # 各エージェントごとの Q テーブル（状態×行動）
        self.Q = [np.zeros((state_dim, n_actions)) for _ in range(n_agents)]
        # 簡易のため状態は離散化せず、そのままインデックスとして使わない（ここではランダム方策のみ）

    def select_actions(self, observations: List[np.ndarray], mode: str) -> List[int]:
        """
        各エージェントの行動を選択（簡易版：ランダム＋ε-greedy風）
        """
        actions = []
        for i in range(self.n_agents):
            # 簡易のためランダム行動（実装練習用）
            if mode == "mode1":
                # 通常の ε-greedy 風
                if np.random.rand() < self.epsilon1:
                    action = np.random.randint(self.n_actions)
                else:
                    # ここでは Q 学習は未実装なのでランダム
                    action = np.random.randint(self.n_actions)
            elif mode == "mode2":
                # Agent 1 が探索重視
                if i == 0 and np.random.rand() < self.epsilon2:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.random.randint(self.n_actions)
            elif mode == "mode3":
                # Agent 2 が探索重視
                if i == 1 and np.random.rand() < self.epsilon2:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.random.randint(self.n_actions)
            else:
                action = np.random.randint(self.n_actions)
            actions.append(action)
        return actions

    def update(self, observations, actions, rewards):
        """
        ここでは簡易のため更新は行わない（Q学習の実装は練習用に任せる）
        """
        pass


def run_episode(env: CooperativeGridEnv, agent: MavenStyleMultiAgent,
                T: int = 20, mode_switch_interval: int = 5,
                record: bool = True) -> Dict:
    """
    1エピソードを実行し、状態・行動・報酬の履歴を返す
    """
    positions_history = []
    observations_history = []
    actions_history = []
    rewards_history = []
    modes_history = []

    modes = ["mode1", "mode2", "mode3"]
    current_mode = np.random.choice(modes)

    agent_positions, observations = env.reset()
    positions_history.append(agent_positions)
    observations_history.append(observations)

    for t in range(T):
        # 一定間隔でモードを切り替え
        if t % mode_switch_interval == 0:
            current_mode = np.random.choice(modes)

        actions = agent.select_actions(observations, current_mode)
        next_positions, next_observations, rewards, done, info = env.step(actions)

        positions_history.append(next_positions)
        observations_history.append(next_observations)
        actions_history.append(actions)
        rewards_history.append(rewards)
        modes_history.append(current_mode)

        observations = next_observations
        if done:
            break

    return {
        "positions": positions_history,
        "observations": observations_history,
        "actions": actions_history,
        "rewards": rewards_history,
        "modes": modes_history,
        "T": len(positions_history),
    }


def visualize_episode(env: CooperativeGridEnv, episode_data: Dict, save_path: Optional[str] = None):
    """
    エピソードの状態遷移を可視化し、必要なら動画として保存
    """
    positions_history = episode_data["positions"]
    T = episode_data["T"]
    grid_size = env.grid_size
    resource_pos = env.resource_pos

    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(frame):
        ax.clear()
        # グリッドを描画
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {frame}/{T-1}")

        # 資源を描画
        rx, ry = resource_pos
        ax.add_patch(plt.Rectangle((rx - 0.5, ry - 0.5), 1, 1, color='gold', alpha=0.7))

        # エージェントを描画
        colors = ['red', 'blue']
        labels = ['Agent 1', 'Agent 2']
        for i, (x, y) in enumerate(positions_history[frame]):
            ax.add_patch(plt.Circle((x, y), 0.3, color=colors[i], alpha=0.8))
            ax.text(x, y, labels[i][-1], ha='center', va='center', fontweight='bold')

        # 前の位置との軌跡を薄く表示（frame>0 のみ）
        if frame > 0:
            for i in range(env.n_agents):
                prev_x, prev_y = positions_history[frame-1][i]
                curr_x, curr_y = positions_history[frame][i]
                ax.plot([prev_x, curr_x], [prev_y, curr_y], color=colors[i], alpha=0.4, linewidth=2)

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=500, repeat=False)

    if save_path:
        # 動画として保存（mp4）
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        anim.save(save_path, writer='ffmpeg', fps=2)
        print(f"動画を保存しました: {save_path}")
    else:
        # Colab 上でアニメーションを表示
        plt.close(fig)
        return HTML(anim.to_jshtml())

    plt.close(fig)
    return anim

