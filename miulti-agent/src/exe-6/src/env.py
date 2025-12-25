import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from typing import Dict, Tuple, List
import random

class SwitchEnv:
    def __init__(self, size=5):
        self.size = size
        self.num_agents = 2
        # ゴール（荷物の目的地）: エージェント0は右端(4)、エージェント1は左端(0)
        self.pickup_pos = 2  # 通路の真ん中
        self.targets = {0: 4, 1: 0} 
        self.reset()

    def reset(self):
        # エージェントの初期位置: 両端
        self.agent_positions = {0: 0, 1: 4}
        self.agent_holding = {0: False, 1: False}
        self.done_delivery = {0: False, 1: False}
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = {}
        for i in range(self.num_agents):
            other = 1 - i
            # (自分の位置, 自分が持っているか, 相方の位置, 相方が持っているか)
            obs[i] = (
                self.agent_positions[i],
                self.agent_holding[i],
                self.agent_positions[other],
                self.agent_holding[other]
            )
        return obs

    def step(self, actions: Dict[int, int]):
        self.steps += 1
        rewards = {i: -0.1 for i in range(self.num_agents)} # 時間経過ペナルティ
        
        # 移動処理 (0: Stay, 1: Left, 2: Right)
        next_pos = {}
        for i, action in actions.items():
            p = self.agent_positions[i]
            if action == 1: p = max(0, p - 1)
            elif action == 2: p = min(self.size - 1, p + 1)
            next_pos[i] = p

        # 衝突判定（同じマスに入ろうとしたら移動失敗）
        if next_pos[0] == next_pos[1]:
            rewards[0] -= 1.0
            rewards[1] -= 1.0
            # 位置は更新しない
        else:
            self.agent_positions = next_pos

        # ピックアップ & ドロップオフ判定
        for i in range(self.num_agents):
            # 1. 荷物を拾う (真ん中にいて、誰も持っていない場合)
            if self.agent_positions[i] == self.pickup_pos and not any(self.agent_holding.values()) and not any(self.done_delivery.values()):
                self.agent_holding[i] = True
                rewards[i] += 5.0
            
            # 2. 届ける (自分のターゲットに到着)
            if self.agent_holding[i] and self.agent_positions[i] == self.targets[i]:
                self.agent_holding[i] = False
                self.done_delivery[i] = True
                rewards[i] += 20.0

        done = {i: any(self.done_delivery.values()) or self.steps >= 50 for i in range(self.num_agents)}
        return self._get_obs(), rewards, done, {}

    def render_frame(self):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks([])
        ax.grid(True)

        # ピックアップ地点
        ax.add_patch(plt.Rectangle((self.pickup_pos-0.4, -0.4), 0.8, 0.8, color='gray', alpha=0.2))
        ax.text(self.pickup_pos, 0.45, "Pickup", ha='center')

        # エージェント描画
        colors = {0: 'red', 1: 'blue'}
        for i in range(self.num_agents):
            pos = self.agent_positions[i]
            color = colors[i]
            marker = 's' if self.agent_holding[i] else 'o'
            ax.plot(pos, 0, marker, markersize=20, color=color, label=f"Agent {i}")
            # ターゲット地点の印
            ax.plot(self.targets[i], 0, 'x', markersize=12, color=color)

        ax.set_title(f"Step: {self.steps}")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    


def save_random_behavior_gif(env, filename="random_behavior.gif", max_steps=30):
    frames = []
    obs = env.reset()
    
    print(f"🎬 Generating random behavior GIF: {filename}...")
    
    for t in range(max_steps):
        # 1. 現在のフレームをキャプチャしてリストに追加
        frame = env.render_frame()
        frames.append(frame)
        
        # 2. 全エージェントに対して完全にランダムな行動を選択
        # 0: Stay, 1: Left, 2: Right
        actions = {i: random.randint(0, 2) for i in range(env.num_agents)}
        
        # 3. 環境を1ステップ進める
        next_obs, rewards, done, info = env.step(actions)
        
        # デバッグ用に報酬を表示（任意）
        # print(f"Step {t}: Actions {actions}, Rewards {rewards}")
        
        if any(done.values()):
            # 終了（ゴールまたはタイムアップ）した場合は最後のフレームを撮って終了
            frames.append(env.render_frame())
            print(f"🏁 Episode finished at step {t}")
            break

    # 4. PIL Imageの機能を使ってGIFとして保存
    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=300,  # ランダムな動きが見やすいよう少しゆっくり（0.3秒間隔）
            loop=0
        )
        print(f"✅ GIF saved successfully: {filename}")
    else:
        print("❌ No frames were captured.")



if __name__ == "__main__":
    # 環境の初期化
    # --- 実行 ---
    env = SwitchEnv(size=5)
    save_random_behavior_gif(env, "switch_random.gif")

