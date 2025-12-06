import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from typing import Dict, Tuple, List
import time

# 環境設定
GRID_SIZE = 15          # マップの広さ
NUM_AGENTS = 3          # ドローンの数
SENSOR_RANGE = 1        # センサーの範囲（1なら自身の周囲1マス=3x3が見える）
MAX_STEPS = 100         # 1エピソードの最大ステップ数

class MultiSensorSearchEnv:
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = 5  # 0:待機, 1:上, 2:下, 3:左, 4:右
        
        # 探索状態マップ (0: 未探索, 1: 探索済み)
        self.explored_map = np.zeros((self.size, self.size), dtype=int)
        
        # Matplotlib用の図の保持
        self.fig = None
        self.ax = None
        
        self.reset()

    def reset(self) -> Dict[int, np.ndarray]:
        """環境を初期化します。"""
        # マップを全て「未探索(0)」にリセット
        self.explored_map = np.zeros((self.size, self.size), dtype=int)
        self.current_step = 0
        
        # エージェントの位置をランダムに初期化（スタート時は重なっても良いとする）
        self.agent_positions = {
            i: (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for i in range(self.num_agents)
        }
        
        # 初期位置での探索を反映
        self._update_exploration()
        
        return self._get_obs()

    def _get_obs(self) -> Dict[int, Dict]:
        """
        観測を返します。
        簡単のため、全エージェントが「自分の位置」と「グローバルな探索マップ」を共有しているとします。
        (分散知識問題にする場合は、ここを個別のローカルマップに変更します)
        """
        obs = {}
        for i in range(self.num_agents):
            obs[i] = {
                "position": self.agent_positions[i],
                "map": self.explored_map.copy() # 共有された探索済みマップ
            }
        return obs

    def _update_exploration(self) -> int:
        """現在のエージェントの位置に基づいて探索済みエリアを更新し、新規探索セル数を返します。"""
        newly_explored_count = 0
        
        for pos in self.agent_positions.values():
            x, y = pos
            # センサー範囲 (正方形) の計算
            x_min = max(0, x - SENSOR_RANGE)
            x_max = min(self.size, x + SENSOR_RANGE + 1)
            y_min = max(0, y - SENSOR_RANGE)
            y_max = min(self.size, y + SENSOR_RANGE + 1)
            
            # 範囲内の未探索セルをカウント
            current_area = self.explored_map[y_min:y_max, x_min:x_max]
            newly_explored_count += np.sum(current_area == 0)
            
            # マップを更新 (1にする)
            self.explored_map[y_min:y_max, x_min:x_max] = 1
            
        return newly_explored_count

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, bool, Dict]:
        """行動を実行し、状態を進めます。"""
        self.current_step += 1
        rewards = {i: -0.1 for i in range(self.num_agents)} # タイムペナルティ
        
        # 1. 移動の実行
        for i, action in actions.items():
            cx, cy = self.agent_positions[i]
            nx, ny = cx, cy

            if action == 1: ny += 1  # 上
            elif action == 2: ny -= 1  # 下
            elif action == 3: nx -= 1  # 左
            elif action == 4: nx += 1  # 右
            
            # マップ外に出ないようにクリップ
            nx = np.clip(nx, 0, self.size - 1)
            ny = np.clip(ny, 0, self.size - 1)
            
            self.agent_positions[i] = (nx, ny)

        # 2. 探索エリアの更新と報酬計算
        # チーム全員で協力して発見した「新しいマス」の数に応じて報酬を与える (協調報酬)
        new_cells = self._update_exploration()
        
        # 協調報酬: 新しく発見したマス1つにつき +1.0 をチーム全員（または発見者）に分配
        # ここではシンプルに全員に同じ報酬を与える「完全協力ゲーム」とします
        team_reward = new_cells * 1.0
        
        for i in range(self.num_agents):
            rewards[i] += team_reward

        # 3. 終了判定
        # 全てのマスが探索済み (探索率 100%) になったら終了
        total_cells = self.size * self.size
        explored_cells = np.sum(self.explored_map)
        coverage_ratio = explored_cells / total_cells
        
        done = coverage_ratio >= 1.0 or self.current_step >= MAX_STEPS
        
        if coverage_ratio >= 1.0:
            # コンプリートボーナス
            for i in range(self.num_agents):
                rewards[i] += 20.0

        info = {
            "coverage": coverage_ratio,
            "new_cells": new_cells
        }

        return self._get_obs(), rewards, done, info

    # --- 可視化メソッド ---
    def render(self, mode='graphic', sleep_time=0.1):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.ax.clear()
        
        # グリッドの描画設定
        # 未探索(0) = 黒/グレー, 探索済み(1) = 白/明るい色
        cmap = colors.ListedColormap(['#333333', '#ffffff']) 
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # マップの表示 (origin='lower'で左下を(0,0)にする)
        self.ax.imshow(self.explored_map, cmap=cmap, norm=norm, origin='lower', extent=[0, self.size, 0, self.size])
        
        # グリッド線
        self.ax.set_xticks(np.arange(0, self.size, 1))
        self.ax.set_yticks(np.arange(0, self.size, 1))
        self.ax.grid(which='both', color='#555555', linestyle='-', linewidth=0.5)
        
        # 進捗率の表示
        coverage = (np.sum(self.explored_map) / (self.size**2)) * 100
        self.ax.set_title(f"Step: {self.current_step} | Coverage: {coverage:.1f}%")

        # エージェントの描画
        agent_colors = ['red', 'cyan', 'yellow', 'lime']
        for i, pos in self.agent_positions.items():
            x, y = pos
            
            # エージェント本体
            circle = patches.Circle((x + 0.5, y + 0.5), 0.3, facecolor=agent_colors[i % len(agent_colors)], edgecolor='black', zorder=10)
            self.ax.add_patch(circle)
            
            # センサー範囲の枠線（オプション）
            rect_size = SENSOR_RANGE * 2 + 1
            rect = patches.Rectangle((x - SENSOR_RANGE, y - SENSOR_RANGE), rect_size, rect_size, 
                                     linewidth=1, edgecolor=agent_colors[i % len(agent_colors)], facecolor='none', linestyle='--', alpha=0.5)
            self.ax.add_patch(rect)

        plt.draw()
        plt.pause(sleep_time)

# --- 実行例 ---
if __name__ == '__main__':
    # ランダムウォークによるシミュレーション
    env = MultiSensorSearchEnv(size=10, num_agents=3) # 10x10マップ, 3機
    obs = env.reset()
    
    print("--- 探索ミッション開始 ---")
    
    try:
        while True:
            # ランダムアクション
            actions = {i: random.randint(0, 4) for i in range(env.num_agents)}
            
            obs, rewards, done, info = env.step(actions)
            
            env.render(sleep_time=0.2)
            
            if done:
                print(f"ミッション終了！ 最終カバレッジ: {info['coverage']*100:.1f}%")
                plt.pause(2.0)
                break
                
    except KeyboardInterrupt:
        print("中断しました")
    
    plt.ioff()
    plt.show()