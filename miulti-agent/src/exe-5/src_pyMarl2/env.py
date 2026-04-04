import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

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
        # エージェントの初期位置（対角からスタート）
        self.agent_pos = np.array([[0, 0], [self.size-1, self.size-1]])
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
        # 1. 行動を適用（ただし衝突やボトルネック制約を考慮）
        new_pos = self.agent_pos.copy()
        for i, a in enumerate(actions):
            if a == 1: new_pos[i][0] = max(0, new_pos[i][0]-1)
            elif a == 2: new_pos[i][0] = min(self.size-1, new_pos[i][0]+1)
            elif a == 3: new_pos[i][1] = max(0, new_pos[i][1]-1)
            elif a == 4: new_pos[i][1] = min(self.size-1, new_pos[i][1]+1)

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

        # 5-1. ターゲットまでの距離に基づく報酬（グローバル）
        dist0 = np.linalg.norm(self.agent_pos[0] - self.targets[0])
        dist1 = np.linalg.norm(self.agent_pos[1] - self.targets[1])
        global_dist_reward = -0.1 * (dist0 + dist1)
        rewards += global_dist_reward

        # 5-2. 個別進捗報酬（停滞を防ぐ）
        prev_dist0 = np.linalg.norm(prev_pos[0] - self.targets[0])
        prev_dist1 = np.linalg.norm(prev_pos[1] - self.targets[1])
        if dist0 < prev_dist0:
            rewards[0] += 0.05  # エージェント0がターゲットに近づいた
        if dist1 < prev_dist1:
            rewards[1] += 0.05  # エージェント1がターゲットに近づいた

        # 5-3. 停滞ペナルティ（同じ位置に留まり続けるとマイナス）
        if np.array_equal(self.agent_pos[0], prev_pos[0]):
            rewards[0] -= 0.1
        if np.array_equal(self.agent_pos[1], prev_pos[1]):
            rewards[1] -= 0.1

        # 5-4. 同時到達ボーナス（協調報酬）を強化
        if dist0 == 0 and dist1 == 0:
            rewards += 10.0  # 両方同時に到達したら大報酬（強化）

        # 5-5. 衝突ペナルティ
        if np.array_equal(self.agent_pos[0], self.agent_pos[1]):
            rewards -= 1.0

        # 5-6. ボトルネック付近での混雑ペナルティ（協調を促す）
        bottleneck_dist0 = np.linalg.norm(self.agent_pos[0] - np.array([self.bottleneck_row, self.bottleneck_cols[0]]))
        bottleneck_dist1 = np.linalg.norm(self.agent_pos[1] - np.array([self.bottleneck_row, self.bottleneck_cols[0]]))
        if bottleneck_dist0 < 2 and bottleneck_dist1 < 2:
            # 両方がボトルネック付近に接近しすぎているとペナルティ
            rewards -= 0.5

        # 5-7. 他エージェントを考慮した協調報酬（相手がボトルネックに近いときに自分が待機／迂回するとボーナス）
        # エージェント0の視点
        if bottleneck_dist1 < 1.5 and dist0 > 1.0:
            # エージェント1がボトルネックに近く、自分はまだ遠い → 待機／迂回を推奨
            rewards[0] += 0.1
        # エージェント1の視点
        if bottleneck_dist0 < 1.5 and dist1 > 1.0:
            # エージェント0がボトルネックに近く、自分はまだ遠い → 待機／迂回を推奨
            rewards[1] += 0.1

        self.steps += 1
        done = (self.steps >= self.max_steps) or (dist0 == 0 and dist1 == 0)
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



class CoopNavWrapper:
    def __init__(self, **kwargs):
        self.env = CooperativeNavigationEnv(**kwargs)
        self.n_agents = self.env.num_agents
        self.n_actions = 5

        self.obs_size = len(self.env._get_obs()[0])
        self.state_size = self.obs_size * self.n_agents

    def reset(self):
        obs = self.env.reset()
        return obs, self.get_state()

    def step(self, actions):
        obs, reward, done, _ = self.env.step(actions)

        # PyMARLはglobal reward前提
        reward = np.mean(reward)

        return reward, done, {}

    def get_obs(self):
        return self.env._get_obs()

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self):
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        pass

    def render(self):
        pass

env = CoopNavWrapper()
env.reset()
env.save_gif()

