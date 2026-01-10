import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

class SimpleReachEnv:
    def __init__(self, size=5):
        self.size = size
        self.num_agents = 2
        self.max_steps = 20
        # ターゲット位置
        self.targets = np.array([[size-1, size-1], [0, 0]]) 
        self.reset()

    def reset(self):
        # エージェントの初期位置
        self.agent_pos = np.array([[0, 0], [self.size-1, self.size-1]])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            # 自分の座標 + ターゲットへの相対距離
            rel_dist = self.targets[i] - self.agent_pos[i]
            obs.append(np.concatenate([self.agent_pos[i]/self.size, rel_dist/self.size]))
        return obs

    def step(self, actions):
        rewards = []
        for i, a in enumerate(actions):
            # 0:待機, 1:上, 2:下, 3:左, 4:右
            if a == 1: self.agent_pos[i][0] = max(0, self.agent_pos[i][0]-1)
            elif a == 2: self.agent_pos[i][0] = min(self.size-1, self.agent_pos[i][0]+1)
            elif a == 3: self.agent_pos[i][1] = max(0, self.agent_pos[i][1]-1)
            elif a == 4: self.agent_pos[i][1] = min(self.size-1, self.agent_pos[i][1]+1)
            
            # 報酬：ターゲットに近づいたらプラス、離れたらマイナス
            dist = np.linalg.norm(self.agent_pos[i] - self.targets[i])
            rewards.append(-dist * 0.1) 
            if dist == 0: rewards[-1] += 1.0 

        self.steps += 1
        done = (self.steps >= self.max_steps) or all([np.array_equal(p, t) for p, t in zip(self.agent_pos, self.targets)])
        return self._get_obs(), rewards, done, {}

    # --- 追加: 可視化メソッド ---
    def render(self, ax):
        """現在の状態をmatplotlibのaxに描画する"""
        ax.clear()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')

        colors = ['red', 'green']
        # ターゲットの描画
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

    def save_gif(self, trainer=None, filename="simple_reach.gif"):
        """エピソードを実行してGIFを保存する (trainerがNoneならランダム)"""
        frames = []
        obs_list = self.reset()
        done = False
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # 初期状態の隠れ状態 (MAPPO/GRU用)
        h_actors = [np.zeros((1, 1, 128)) for _ in range(self.num_agents)]

        while not done:
            # フレームを描画
            self.render(ax)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))

            # 行動選択
            if trainer is None:
                # ランダム行動
                actions = [np.random.randint(0, 5) for _ in range(self.num_agents)]
            else:
                # 学習済みエージェントの行動 (以前作成したtrainerのロジックを使用)
                import torch
                obs_tensor = trainer.normalize_obs(obs_list)
                actions = []
                for i in range(self.num_agents):
                    agent_id = torch.zeros(self.num_agents); agent_id[i] = 1.0
                    inp = torch.cat([obs_tensor[i], agent_id], dim=-1).view(1, 1, -1)
                    with torch.no_grad():
                        # ここでは決定論的な行動 (argmax) を選択
                        # probs, _ = trainer.actor(inp, torch.FloatTensor(h_actors[i]))
                        # actions.append(torch.argmax(probs).item())
                        dist, _ = trainer.actor(inp, torch.FloatTensor(h_actors[i]))
                        # Categoricalオブジェクト(dist)から、実際の確率(probs)を取り出してargmaxをとる
                        actions.append(torch.argmax(dist.probs).item())
                
            obs_list, _, done, _ = self.step(actions)

        # GIFの生成
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=300, loop=0)
        plt.close(fig)
        print(f"✅ GIF saved as {filename}")