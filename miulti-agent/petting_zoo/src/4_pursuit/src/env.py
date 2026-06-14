import numpy as np
from pettingzoo.sisl import pursuit_v4

class PursuitWrapper:
    def __init__(self, render_mode=None, max_cycles=500, obs_range=7):
        self.env = pursuit_v4.env(
            render_mode=render_mode,
            max_cycles=max_cycles,
            shared_reward=True
        )
        self.obs_range = obs_range
        self.center_idx = obs_range // 2  # 7x7 の中心 (3, 3) が自分自身の位置
        
        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)
        
        self.obs_dim = (obs_range * obs_range * 3)
        self.state_dim = self.obs_dim * self.num_agents
        
        self.action_space = self.env.action_space(self.possible_agents[0])
        self.action_dim = self.action_space.n

        # --- 報酬系のスケール設定 ---
        self.distance_reward_scale = 0.05
        self.coop_reward_scale = 0.1
        self.surround_reward = 1.0
        
        # 新設：味方のいないルートからアプローチしたときのボーナス
        self.flanking_bonus_scale = 0.1  

        # エージェントごとの前ステップの「最も近い獲物への距離」を記録する辞書
        self.prev_min_distances = {}

    def reset(self):
        self.env.reset()
        self.prev_min_distances = {agent: None for agent in self.possible_agents}

    def get_obs(self, agent):
        if agent not in self.env.agents:
            return None
        obs, _, _, _, _ = self.env.last(agent)
        if obs is None:
            return None
        return obs

    def get_global_state(self):
        obs_list = []
        for agent in self.possible_agents:
            obs = self.get_obs(agent)
            if obs is not None:
                obs_list.append(obs.reshape(-1).astype(np.float32))
            else:
                obs_list.append(np.zeros(self.obs_dim, dtype=np.float32))
        return np.concatenate(obs_list)

    def _analyze_observation(self, obs):
        """
        1つのエージェントの観測(7, 7, 3)から報酬計算用の情報を抽出する
        """
        if obs is None:
            return None, 0, 0

        prey_layer = obs[:, :, 2]
        ally_layer = obs[:, :, 1]
        
        prey_positions = np.argwhere(prey_layer > 0)

        min_dist = float('inf')
        closest_prey_pos = None
        cy, cx = self.center_idx, self.center_idx

        # 1. 最も近い獲物を特定
        for py, px in prey_positions:
            dist = abs(py - cy) + abs(px - cx)
            if dist < min_dist:
                min_dist = dist
                closest_prey_pos = (py, px)

        if min_dist == float('inf'):
            return None, 0, 0  # 視界内に獲物がいない場合

        # 2. 協調判定：自分の隣接4マスにいる味方の総数
        allies_count = 0
        neighbors = [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]
        for ny, nx in neighbors:
            if 0 <= ny < self.obs_range and 0 <= nx < self.obs_range:
                allies_count += ally_layer[ny, nx]

        # 3. 新設：獲物がいる「方向（セクター）」に他の味方が何人いるかをカウント
        # 獲物の位置に応じて、自分の視界を4つのエリア（象限）に分けて味方を数える
        py, px = closest_prey_pos
        flank_allies = 0
        
        # 獲物が自分より「上」のエリアにいる場合、上半分(y <= 3)の味方をチェック
        if py < cy:
            flank_allies += np.sum(ally_layer[0:cy, :])
        # 獲物が自分より「下」のエリアにいる場合、下半分(y >= 3)の味方をチェック
        elif py > cy:
            flank_allies += np.sum(ally_layer[cy+1:, :])
            
        # 獲物が自分より「左」のエリアにいる場合、左半分(x <= 3)の味方をチェック
        if px < cx:
            flank_allies += np.sum(ally_layer[:, 0:cx])
        # 獲物が自分より「右」のエリアにいる場合、右半分(x >= 3)の味方をチェック
        elif px > cx:
            flank_allies += np.sum(ally_layer[:, cx+1:])

        return min_dist, allies_count, flank_allies

    def step(self, agent, action):
        if agent not in self.env.agents:
            return 0.0, True, True, {}
        
        _, _, terminated, truncated, _ = self.env.last(agent)
        step_action = None if (terminated or truncated) else action
        
        self.env.step(step_action)
        
        if agent not in self.env.agents:
            return 0.0, True, True, {}
        
        obs, reward, terminated, truncated, info = self.env.last(agent)

        # 観測データから距離、周囲の味方数、同じ方向にいる味方数を抽出
        current_min_dist, allies_count, flank_allies = self._analyze_observation(obs)

        # 1. 距離ベース報酬の計算
        reward_distance = 0.0
        prev_dist = self.prev_min_distances.get(agent)
        
        if current_min_dist is not None and prev_dist is not None:
            change = prev_dist - current_min_dist
            reward_distance = change * self.distance_reward_scale
            
            # 【新設】包囲（側面攻撃）ボーナス
            # 獲物に近づいた（change > 0）かつ、「その方向に味方が少ない（例: 0人）」場合に追加ボーナス
            if change > 0 and flank_allies == 0:
                reward_distance += self.flanking_bonus_scale

        self.prev_min_distances[agent] = current_min_dist

        # 2. 協調行動報酬の計算
        reward_coop = 0.0
        if current_min_dist is not None and current_min_dist <= 2:
            if allies_count >= 1:
                reward_coop += self.coop_reward_scale
            if allies_count >= 3:
                reward_coop += self.surround_reward

        # 報酬を加算
        reward += reward_distance + reward_coop

        return reward, terminated, truncated, info