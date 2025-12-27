import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, List
import time
import numpy as np
from IPython import display # Jupyter環境用（オプション）

# 環境設定
GRID_SIZE = 10
NUM_AGENTS = 2
NUM_ORDERS = 3
PICKUP_LOCATIONS = [(1, 1), (8, 1), (5, 8)]
DROPOFF_LOCATION = (5, 5)

class WarehouseEnv:
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = 5  # 0:待機, 1:上, 2:下, 3:左, 4:右
        
        # Matplotlib用の図の保持
        self.fig = None
        self.ax = None
        
        # 状態の初期化
        self.reset()

    def reset(self) -> Dict[int, Tuple]:
        """環境を初期化し、初期状態を返します。"""
        self.agent_positions: Dict[int, Tuple[int, int]] = {
            i: (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for i in range(self.num_agents)
        }
        self.agent_holding: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.remaining_orders: List[int] = list(range(NUM_ORDERS))
        return self._get_obs()

    def _get_obs(self) -> Dict[int, Tuple]:
        obs = {}
        for i in range(self.num_agents):
            obs[i] = (
                self.agent_positions[i],
                self.agent_holding[i],
                tuple(self.remaining_orders)
            )
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        next_positions: Dict[int, Tuple[int, int]] = {}
        rewards: Dict[int, float] = {i: 0.0 for i in range(self.num_agents)}

        # --- 距離ベース報酬のための事前計算 ---
        old_distances = {}
        for i in range(self.num_agents):
            old_distances[i] = self._get_distance_to_target(i)

        # 1. 位置の更新
        for i, action in actions.items():
            current_x, current_y = self.agent_positions[i]
            next_x, next_y = current_x, current_y
            if action == 1: next_y += 1  # 上
            elif action == 2: next_y -= 1  # 下
            elif action == 3: next_x -= 1  # 左
            elif action == 4: next_x += 1  # 右
            next_x = np.clip(next_x, 0, self.size - 1)
            next_y = np.clip(next_y, 0, self.size - 1)
            next_positions[i] = (next_x, next_y)

        # 2. 衝突判定 (変更なし)
        final_positions = self.agent_positions.copy()
        is_collision = False
        for i in range(self.num_agents):
            pos = next_positions[i]
            is_colliding = any(i != j and pos == next_positions[j] for j in range(self.num_agents))
            if is_colliding:
                rewards[i] -= 5.0  # 衝突ペナルティ
                is_collision = True
            else:
                final_positions[i] = pos
        self.agent_positions = final_positions

        # 3. 距離の変化による報酬 (Reward Shaping)
        for i in range(self.num_agents):
            new_dist = self._get_distance_to_target(i)
            # 近づいたら報酬、離れたらペナルティ
            if new_dist < old_distances[i]:
                rewards[i] += 0.1
            elif new_dist > old_distances[i]:
                rewards[i] -= 0.1
            
            # 時間経過の微小ペナルティ
            rewards[i] -= 0.05

        # 4. ピックアップ・ドロップオフ (イベント報酬)
        for i in range(self.num_agents):
            current_pos = self.agent_positions[i]
            if not self.agent_holding[i]:
                for order_idx in list(self.remaining_orders):
                    if current_pos == PICKUP_LOCATIONS[order_idx]:
                        self.agent_holding[i] = True
                        self.remaining_orders.remove(order_idx)
                        rewards[i] += 10.0  # ピックアップ報酬
                        break
            elif self.agent_holding[i]:
                if current_pos == DROPOFF_LOCATION:
                    self.agent_holding[i] = False
                    rewards[i] += 50.0  # ドロップオフ報酬
                    
        done = {i: len(self.remaining_orders) == 0 and not any(self.agent_holding.values()) 
                for i in range(self.num_agents)}
        return self._get_obs(), rewards, done, {"collision": is_collision}

    def _get_distance_to_target(self, agent_idx: int) -> float:
        """エージェントから現在のターゲット（荷物または配送先）への最短マンハッタン距離を返します。"""
        curr_pos = self.agent_positions[agent_idx]
        
        if self.agent_holding[agent_idx]:
            # 荷物を持っているならドロップオフ地点がターゲット
            target = DROPOFF_LOCATION
            return abs(curr_pos[0] - target[0]) + abs(curr_pos[1] - target[1])
        else:
            # 荷物を持っていないなら、残っている荷物の中で最も近いものがターゲット
            if not self.remaining_orders:
                return 0.0
            
            distances = []
            for order_idx in self.remaining_orders:
                target = PICKUP_LOCATIONS[order_idx]
                dist = abs(curr_pos[0] - target[0]) + abs(curr_pos[1] - target[1])
                distances.append(dist)
            return min(distances)
        
    def _render_graphic(self, sleep_time):
        if self.fig is None:
            plt.ion() # インタラクティブモードON
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True)
        self.ax.set_title(f"Orders Remaining: {len(self.remaining_orders)}")

        # ドロップオフ地点 (赤色)
        dx, dy = DROPOFF_LOCATION
        self.ax.add_patch(patches.Rectangle((dx-0.5, dy-0.5), 1, 1, color='red', alpha=0.3, label='Dropoff'))
        self.ax.text(dx, dy, 'Drop', ha='center', va='center', fontsize=8, color='darkred')

        # ピックアップ地点 (青色)
        for idx in self.remaining_orders:
            px, py = PICKUP_LOCATIONS[idx]
            self.ax.add_patch(patches.Rectangle((px-0.5, py-0.5), 1, 1, color='blue', alpha=0.3, label='Pickup'))
            self.ax.text(px, py, 'Pick', ha='center', va='center', fontsize=8, color='darkblue')

        # エージェント (円)
        colors = ['green', 'orange', 'purple', 'cyan']
        for i, pos in self.agent_positions.items():
            ax, ay = pos
            color = colors[i % len(colors)]
            # 荷物を持っている場合は枠線を太く、色を変えるなどの表現
            edgecolor = 'black'
            linewidth = 1
            if self.agent_holding[i]:
                linewidth = 3
                edgecolor = 'red' # 荷物持ち強調
            
            circle = patches.Circle((ax, ay), 0.3, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, label=f'Agent {i}')
            self.ax.add_patch(circle)
            self.ax.text(ax, ay, f'A{i}', ha='center', va='center', color='white', fontweight='bold')

        plt.draw()
        plt.pause(sleep_time)