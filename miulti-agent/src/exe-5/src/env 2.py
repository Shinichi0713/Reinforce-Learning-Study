import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import torch

class CooperativeNavigationEnv:
    def __init__(self, size=5):
        self.size = size
        self.num_agents = 2
        self.max_steps = 30  # 少し長めに設定

        # ターゲット位置（両方同じターゲットに設定 → 協調タスク化）
        self.targets = np.array([[size-1, size-1], [size-1, size-1]]) 

        # 狭い通路（ボトルネック）の位置（中央の1マス幅）
        self.bottleneck_row = size // 2
        self.bottleneck_cols = [size // 2]  # 中央1マスのみ通行可能

        self.reset()

    def reset(self):
        # ランダムな初期位置を生成（衝突せず、ボトルネックにも乗らない）
        while True:
            pos0 = np.random.randint(0, self.size, size=2)
            pos1 = np.random.randint(0, self.size, size=2)
            
            # 条件チェック
            # 1. 両エージェントが同じセルでない
            if np.array_equal(pos0, pos1):
                continue
            # 2. どちらもボトルネックセルでない（中央行の中央列）
            if (pos0[0] == self.bottleneck_row and pos0[1] == self.bottleneck_cols[0]) or \
              (pos1[0] == self.bottleneck_row and pos1[1] == self.bottleneck_cols[0]):
                continue
            # 3. どちらもターゲット位置（右下）でない（到達済み状態を避ける）
            if np.array_equal(pos0, self.targets[0]) or np.array_equal(pos1, self.targets[0]):
                continue
            
            # 条件を満たしたら採用
            self.agent_pos = np.array([pos0, pos1])
            break

        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            # 自分の位置・ターゲット相対距離
            rel_dist = self.targets[i] - self.agent_pos[i]
            # 他エージェントとの相対位置
            other_pos = self.agent_pos[1 - i] - self.agent_pos[i]
            # ボトルネックまでの相対位置（協調に有用な情報）
            bottleneck_rel = np.array([self.bottleneck_row, self.bottleneck_cols[0]]) - self.agent_pos[i]
            obs.append(np.concatenate([
                self.agent_pos[i] / self.size,
                rel_dist / self.size,
                other_pos / self.size,
                bottleneck_rel / self.size
            ]))
        return obs

    def step(self, actions):
        # ゴール到達済みかチェック
        dist0 = np.linalg.norm(self.agent_pos[0] - self.targets[0])
        dist1 = np.linalg.norm(self.agent_pos[1] - self.targets[1])
        is_goal = (dist0 == 0 and dist1 == 0)

        if is_goal:
            # ゴール到達済みなら、行動を無視して同じ位置に留める
            rewards = [0.0, 0.0]  # 報酬は0（もしくはゴール報酬を1回だけ与える）
            done = True
            return self._get_obs(), rewards, done, {}
        # 1. 行動を適用（ただし衝突やボトルネック制約を考慮）
        new_pos = self.agent_pos.copy()
        for i, a in enumerate(actions):
            if a == 0: new_pos[i][0] = max(0, new_pos[i][0]-1)      # 上
            elif a == 1: new_pos[i][0] = min(self.size-1, new_pos[i][0]+1)  # 下
            elif a == 2: new_pos[i][1] = max(0, new_pos[i][1]-1)      # 左
            elif a == 3: new_pos[i][1] = min(self.size-1, new_pos[i][1]+1)  # 右

        # 2. ボトルネック制約：中央行の特定列以外は通れない
        for i in range(self.num_agents):
            r, c = new_pos[i]
            if r == self.bottleneck_row and c not in self.bottleneck_cols:
                # ボトルネック以外のセルには移動できない（元の位置に留まる）
                new_pos[i] = self.agent_pos[i].copy()

        # 3. 衝突チェック：同じセルには同時に入れない
        if np.array_equal(new_pos[0], new_pos[1]):
            # 衝突した場合は両方とも元の位置に戻す
            new_pos = self.agent_pos.copy()

        # 4. 前回位置との比較（停滞チェック用）
        prev_pos = self.agent_pos.copy()
        self.agent_pos = new_pos

        # 5. 報酬計算（協調ナビゲーション用）
        rewards = np.zeros(self.num_agents, dtype=float)

        # 1. 距離報酬を強化（グローバル）
        dist0 = np.linalg.norm(self.agent_pos[0] - self.targets[0])
        dist1 = np.linalg.norm(self.agent_pos[1] - self.targets[1])
        global_dist_reward = -0.2 * (dist0 + dist1)  # 係数を少し大きく
        rewards += global_dist_reward

        # 2. 個別進捗報酬を強化
        prev_dist0 = np.linalg.norm(prev_pos[0] - self.targets[0])
        prev_dist1 = np.linalg.norm(prev_pos[1] - self.targets[1])
        if dist0 < prev_dist0:
            rewards[0] += 0.2  # 大きくする
        if dist1 < prev_dist1:
            rewards[1] += 0.2

        # 3. 停滞ペナルティは維持
        if np.array_equal(self.agent_pos[0], prev_pos[0]):
            rewards[0] -= 0.4
        if np.array_equal(self.agent_pos[1], prev_pos[1]):
            rewards[1] -= 0.4

        # ゴール判定
        dist0 = np.linalg.norm(self.agent_pos[0] - self.targets[0])
        dist1 = np.linalg.norm(self.agent_pos[1] - self.targets[1])
        is_goal = (dist0 == 0 and dist1 == 0)

        # ゴール報酬を大きくし、ゴール到達で即終了
        if is_goal:
            rewards += 30000.0  # ゴール報酬を強化
            done = True
            return self._get_obs(), rewards.tolist(), done, {}

        # ゴール未到達の場合のみ、距離報酬・停滞ペナルティなどを計算
        global_dist_reward = -0.2 * (dist0 + dist1)
        rewards += global_dist_reward

        # ...（個別進捗報酬・停滞ペナルティなど）...

        self.steps += 1
        done = (self.steps >= self.max_steps)
        return self._get_obs(), rewards.tolist(), done, {}

    # --- 可視化メソッド（ボトルネックも描画） ---
    def render(self, ax):
        ax.clear()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')

        # ボトルネックの描画（通行不可セルを灰色で表示）
        for c in range(self.size):
            if c not in self.bottleneck_cols:
                rect = patches.Rectangle((c, self.size - 1 - self.bottleneck_row), 1, 1,
                                         color='gray', alpha=0.3)
                ax.add_patch(rect)

        colors = ['red', 'green']
        # ターゲットの描画（両方同じ位置）
        for i, (tx, ty) in enumerate(self.targets):
            circle = patches.Circle((ty + 0.5, self.size - 1 - tx + 0.5), 0.3, 
                                    color=colors[i], alpha=0.2)
            ax.add_patch(circle)
            ax.text(ty + 0.3, self.size - 1 - tx + 0.3, f"T{i}", color=colors[i], fontsize=10)

        # エージェントの描画
        for i, (ax_p, ay_p) in enumerate(self.agent_pos):
            rect = patches.Rectangle((ay_p + 0.1, self.size - 1 - ax_p + 0.1), 0.8, 0.8, 
                                     color=colors[i], alpha=0.8)
            ax.add_patch(rect)
            ax.text(ay_p + 0.3, self.size - 1 - ax_p + 0.3, f"A{i}", color='white', weight='bold')

        ax.set_title(f"Step: {self.steps}")

    def save_gif(self, trainer=None, filename="coop_nav_qmix.gif"):
      """QMIX用：学習済みモデルでGIFを保存する"""
      frames = []
      obs_list = self.reset()
      done = False
      
      fig, ax = plt.subplots(figsize=(5, 5))
      
      while not done:
          self.render(ax)
          buf = io.BytesIO()
          plt.savefig(buf, format='png', bbox_inches='tight')
          buf.seek(0)
          frames.append(Image.open(buf))

          if trainer is None:
              # ランダム行動
              actions = [np.random.randint(0, 5) for _ in range(self.num_agents)]
          else:
              # QMIXモデルから行動を選択（決定論的：argmax）
              obs_tensor = trainer.normalize_obs(obs_list)  # (num_agents, obs_dim)
              actions = []
              for i in range(self.num_agents):
                  with torch.no_grad():
                      q_values = trainer.q_nets[i](obs_tensor[i].unsqueeze(0))  # (1, action_dim)
                      action = q_values.argmax().item()
                  actions.append(action)
          
          obs_list, _, done, _ = self.step(actions)

      frames[0].save(filename, save_all=True, append_images=frames[1:], duration=300, loop=0)
      plt.close(fig)
      print(f"✅ GIF saved as {filename}")

