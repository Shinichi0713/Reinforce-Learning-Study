import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image

# ==============================
# Multi-Agent Delivery Environment with GIF Save
# ==============================
class DroneDeliveryEnv:
    def __init__(self, grid_size=10, num_agents=2, num_packages=3, max_steps=200):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_packages = num_packages
        self.max_steps = max_steps
        self.agent_pos = None
        self.agent_has = None
        self.packages = None
        self.step_count = 0
        self.fig = None
        self.ax = None

    def reset(self):
        self.step_count = 0
        self.agent_pos = [self._random_empty_cell([]) for _ in range(self.num_agents)]
        self.agent_has = [-1 for _ in range(self.num_agents)]
        self.packages = []
        used = self.agent_pos.copy()
        for _ in range(self.num_packages):
            pick = self._random_empty_cell(used)
            used.append(pick)
            drop = self._random_empty_cell(used)
            used.append(drop)
            self.packages.append([pick, drop, False, False])
        return self._get_obs()

    def _random_empty_cell(self, occupied):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in occupied:
                return pos

    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            agent_state = {
                "agent_pos": self.agent_pos[i],
                "carrying": self.agent_has[i],
                "packages": self.packages,
                "other_agent": self.agent_pos[1-i],
            }
            obs.append(agent_state)
        return obs

    def step(self, actions):
        rewards = [0, 0]
        done = False
        for i in range(self.num_agents):
            a = actions[i]
            x, y = self.agent_pos[i]
            if a == 1: x = max(0, x - 1)
            elif a == 2: x = min(self.grid_size - 1, x + 1)
            elif a == 3: y = max(0, y - 1)
            elif a == 4: y = min(self.grid_size - 1, y + 1)
            self.agent_pos[i] = (x, y)

        if self.agent_pos[0] == self.agent_pos[1]:
            rewards[0] -= 5; rewards[1] -= 5

        for i in range(self.num_agents):
            pos = self.agent_pos[i]
            carry = self.agent_has[i]
            action = actions[i]
            if action == 5 and carry == -1:
                for pid, pack in enumerate(self.packages):
                    pick, drop, picked, delivered = pack
                    if not picked and pos == pick:
                        self.agent_has[i] = pid
                        pack[2] = True
                        rewards[i] += 1
                        break
            if action == 6 and carry != -1:
                pid = carry
                pick, drop, picked, delivered = self.packages[pid]
                if pos == drop and picked and not delivered:
                    self.agent_has[i] = -1
                    self.packages[pid][3] = True
                    rewards[i] += 10

        if all(p[3] for p in self.packages):
            done = True
            rewards = [r + 5 for r in rewards]

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return self._get_obs(), rewards, done, {}

    # -----------------------------
    # Render for GUI
    # -----------------------------
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self._draw_frame()
        plt.pause(0.01)

    def _draw_frame(self):
        self.ax.clear()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.ax.add_patch(patches.Rectangle((y, self.grid_size-1-x), 1, 1, fill=False, edgecolor='gray'))
        for pid, (pick, drop, picked, delivered) in enumerate(self.packages):
            px, py = (pick[1], self.grid_size - 1 - pick[0])
            dx, dy = (drop[1], self.grid_size - 1 - drop[0])
            if not picked: self.ax.add_patch(patches.Circle((px+0.5, py+0.5), 0.3, color="red"))
            if not delivered: self.ax.add_patch(patches.Circle((dx+0.5, dy+0.5), 0.3, color="green"))
        colors = ["blue", "orange"]
        for i in range(self.num_agents):
            x, y = self.agent_pos[i]
            cx, cy = y, self.grid_size - 1 - x
            self.ax.add_patch(patches.Rectangle((cx, cy), 1, 1, color=colors[i], alpha=0.8))
            if self.agent_has[i] != -1:
                self.ax.text(cx+0.3, cy+0.3, "P", color="white", fontsize=12)
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Step: {self.step_count}")

    # -----------------------------
    # New: GIF Save Method
    # -----------------------------
    def save_gif(self, agent_model=None, filename="delivery_task.gif", max_steps=100):
        frames = []
        self.reset()
        
        # 保存用に専用のFigureを作成
        fig_save, ax_save = plt.subplots(figsize=(5, 5))
        self.fig, self.ax = fig_save, ax_save
        
        print(f"🎬 Recording {filename}...")
        
        for t in range(max_steps):
            # 現在のフレームを描画
            self._draw_frame()
            
            # Figureを画像バッファに変換
            buf = io.BytesIO()
            fig_save.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))
            
            # 行動の決定
            if agent_model is None:
                actions = [np.random.randint(0, 7) for _ in range(self.num_agents)]
            else:
                # 学習済みモデルがある場合はここをモデルの推論処理に書き換える
                obs = self._get_obs()
                actions = agent_model.get_action(obs) 
            
            obs, rew, done, info = self.step(actions)
            if done: break
        
        # GIFとして書き出し
        if frames:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=200, # 1フレームあたりの時間(ms)
                loop=0
            )
            print(f"✅ Saved GIF to {filename}")
        
        plt.close(fig_save)
        self.fig, self.ax = None, None # リセット

# ==============================
# 実行例 (GIF保存)
# ==============================
if __name__ == "__main__":
    env = DroneDeliveryEnv(grid_size=10, num_agents=2, num_packages=3)
    
    # モデルなし(ランダム)で50ステップ分保存
    env.save_gif(filename="drone_marl_test.gif", max_steps=50)

    # Google Colabなどで表示する場合
    try:
        from IPython.display import Image as DisplayImage
        display(DisplayImage(open("drone_marl_test.gif", "rb").read()))
    except ImportError:
        pass