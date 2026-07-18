import numpy as np
from pettingzoo.sisl import pursuit_v4

class PursuitWrapper:
    def __init__(self, render_mode=None, max_cycles=500, obs_range=7):
        # チームとしての共通目的を維持するため shared_reward=True で環境を初期化
        self.env = pursuit_v4.env(
            render_mode=render_mode,
            max_cycles=max_cycles,
            shared_reward=True
        )
        self.obs_range = obs_range
        self.center_idx = obs_range // 2  # 7x7 の中心 (3, 3) が自分自身の位置

        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)

        # 空間観測: 7x7x4 = 196次元 / 味方の行動履歴: 5行動 × 8人 = 40次元 -> 計236次元
        self.spatial_dim = (obs_range * obs_range * 4)
        self.action_history_dim = 5 * self.num_agents
        self.obs_dim = self.spatial_dim + self.action_history_dim
        self.state_dim = self.obs_dim * self.num_agents

        self.action_space = self.env.action_space(self.possible_agents[0])
        self.action_dim = self.action_space.n

        # ハイブリッド報酬の重みパラメータ
        self.distance_reward_scale = 0.005  # 0.01*0 から固定値にする場合は調整してください
        self.coop_reward_scale = 0.02          # 獲物の近くで味方と連携したときの報酬
        self.flanking_bonus_scale = 0.1       # 他に味方がいないルートから回り込んだときの報酬
        self.surround_reward = 50.0            # 3人以上で包囲網を形成したときの報酬

        # 包囲網シェイピング用のパラメータ
        self.soft_gather_reward_scale = 0.05    # 2マス先までに味方が集まっているとき
        self.cross_position_reward_scale = 0.25 # 上下左右のジャスト位置に配置されたとき

        # 連携人数最適化パラメータ
        self.optimal_coop_scale = 0.1          # 4人連携時のベースボーナス
        self.overcrowd_penalty_scale = -0.05    # 5人以上で群がったときのペナルティ（減衰用）

        # 同時アプローチボーナス
        self.simultaneous_approach_bonus = 0.5  # 4人が同時に敵に近づいたときのボーナス

        # 🌟 新設: 索敵フェーズ用の報酬パラメータ
        self.search_coop_move_bonus = 0.05      # 敵がいない時、4人チームで移動した時のボーナス
        self.search_stagnation_penalty = -0.1   # 敵がいない時、その場に留まり続けた（うろうろ含む）ペナルティ

        # 衝突ペナルティ
        self.collision_penalty = -0.5

        # エージェントごとの状態記録
        self.prev_min_distances = {}
        self.prev_agent_positions = {}         # 🌟 追加: 前ステップのグローバル座標記録用
        self.capture_count = 0
        self.captured_prey_ids = set()

        # 全エージェントの直前行動バッファ
        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
        self.last_actions[:, 4] = 1.0  # 4: 滞在 (NONE)

    def reset(self):
        self.env.reset()
        self.prev_min_distances = {agent: None for agent in self.possible_agents}
        self.prev_agent_positions = {agent: None for agent in self.possible_agents} # 🌟 初期化
        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
        self.last_actions[:, 4] = 1.0
        self.capture_count = 0
        self.captured_prey_ids = set()

    def get_obs(self, agent):
        if agent not in self.env.agents:
            return None
        obs, _, _, _, _ = self.env.last(agent)
        if obs is None:
            return None

        wall_layer = obs[:, :, 0]
        ally_layer = obs[:, :, 1]
        prey_layer = obs[:, :, 2]

        id_ally_layer = np.zeros((self.obs_range, self.obs_range), dtype=np.float32)
        raw_env = self.env.unwrapped
        try:
            my_agent_obj = next(a for a in raw_env.agents if a.name == agent)
            my_y, my_x = my_agent_obj.state[1], my_agent_obj.state[0]

            for other_agent in raw_env.agents:
                if other_agent.name == agent:
                    continue
                oy, ox = other_agent.state[1], other_agent.state[0]
                local_y = oy - my_y + self.center_idx
                local_x = ox - my_x + self.center_idx

                if 0 <= local_y < self.obs_range and 0 <= local_x < self.obs_range:
                    agent_id = int(other_agent.name.split('_')[-1]) + 1
                    id_ally_layer[local_y, local_x] = float(agent_id)
        except Exception:
            id_ally_layer = ally_layer.astype(np.float32)

        empty_layer = ((wall_layer == 0) & (id_ally_layer == 0) & (prey_layer == 0)).astype(np.float32)

        semantic_obs = np.stack([
            empty_layer,
            wall_layer.astype(np.float32),
            id_ally_layer,
            prey_layer.astype(np.float32)
        ], axis=-1)

        spatial_flat = semantic_obs.reshape(-1)
        action_history_flat = self.last_actions.reshape(-1)
        full_obs = np.concatenate([spatial_flat, action_history_flat])
        return full_obs

    def get_global_state(self):
        obs_list = []
        for agent in self.possible_agents:
            obs_flat = self.get_obs(agent)
            if obs_flat is not None:
                obs_list.append(obs_flat)
            else:
                obs_list.append(np.zeros(self.obs_dim, dtype=np.float32))
        return np.concatenate(obs_list)

    def _local_to_global(self, agent, local_y, local_x):
        raw_env = self.env.unwrapped
        try:
            my_agent_obj = next(a for a in raw_env.agents if a.name == agent)
            my_y, my_x = my_agent_obj.state[1], my_agent_obj.state[0]
            offset_y = local_y - self.center_idx
            offset_x = local_x - self.center_idx
            return (my_y + offset_y, my_x + offset_x)
        except Exception:
            current_cycle = getattr(raw_env, 'cycles', 0)
            return (current_cycle, local_y, local_x)

    def count_captures(self, agent, prey_positions, ally_layer):
        count_capture = 0
        for py, px in prey_positions:
            allies_around_prey = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.obs_range and 0 <= nx < self.obs_range:
                        if ally_layer[ny, nx] > 0:
                            allies_around_prey += 1

            if allies_around_prey >= 3:
                prey_id = self._local_to_global(agent, py, px)
                if prey_id not in self.captured_prey_ids:
                    self.captured_prey_ids.add(prey_id)
                    self.capture_count += 1
                    count_capture += 1
        return count_capture

    def _analyze_observation(self, agent, obs_flat):
        if obs_flat is None:
            return None, 0, 0, 0.0, 0, 0.0

        spatial_part = obs_flat[:self.spatial_dim]
        obs = spatial_part.reshape(self.obs_range, self.obs_range, 4)

        ally_layer = obs[:, :, 2]
        prey_layer = obs[:, :, 3]

        prey_positions = np.argwhere(prey_layer > 0)
        count_capture = self.count_captures(agent, prey_positions, ally_layer)

        min_dist = float('inf')
        closest_prey_pos = None
        cy, cx = self.center_idx, self.center_idx

        for py, px in prey_positions:
            dist = abs(py - cy) + abs(px - cx)
            if dist < min_dist:
                min_dist = dist
                closest_prey_pos = (py, px)

        if min_dist == float('inf'):
            return None, 0, 0, 0.0, count_capture, 0.0

        # 隣接マスの味方カウント
        allies_count = 0
        neighbors = [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]
        for ny, nx in neighbors:
            if 0 <= ny < self.obs_range and 0 <= nx < self.obs_range:
                if ally_layer[ny, nx] > 0:
                    allies_count += 1

        # 回り込み判定用
        py, px = closest_prey_pos
        flank_allies = 0
        if py < cy: flank_allies += np.sum(ally_layer[0:cy, :] > 0)
        elif py > cy: flank_allies += np.sum(ally_layer[cy+1:, :] > 0)
        if px < cx: flank_allies += np.sum(ally_layer[:, 0:cx] > 0)
        elif px > cx: flank_allies += np.sum(ally_layer[:, cx+1:] > 0)

        # 通常のシェイピング報酬および連携人数のカウント
        shaping_reward = 0.0
        total_allies_around_closest_prey = 0  # 最も近い獲物の周囲2マスにいる味方の総数

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                target_y = py + dy
                target_x = px + dx

                if (0 <= target_y < self.obs_range) and (0 <= target_x < self.obs_range) and (dy != 0 or dx != 0):
                    if ally_layer[target_y, target_x] > 0:
                        total_allies_around_closest_prey += 1
                        m_dist = abs(dy) + abs(dx)
                        if m_dist == 1:
                            shaping_reward += self.cross_position_reward_scale
                        elif m_dist <= 2:
                            shaping_reward += self.soft_gather_reward_scale

        # 自分自身もカウントに含める（視野の中心にいるため、敵の2マス以内に自分がいる場合）
        if min_dist <= 2:
            total_allies_around_closest_prey += 1

        # 4人をベスト（ピーク）とし、離れるほどペナルティを与えるロジック
        coop_density_reward = 0.0
        discrepancy = abs(total_allies_around_closest_prey - 4)
        
        if discrepancy == 0:
            coop_density_reward += self.optimal_coop_scale
        else:
            coop_density_reward += discrepancy * abs(self.overcrowd_penalty_scale) * -1.0

        return min_dist, allies_count, flank_allies, shaping_reward, count_capture, coop_density_reward

    def step(self, agent, action):
        if agent not in self.env.agents:
            return np.zeros(self.obs_dim, dtype=np.float32), 0.0, True, True, {}, 0

        current_cycle = getattr(self.env.unwrapped, 'cycles', 0)
        _, _, terminated, truncated, _ = self.env.last(agent)
        step_action = None if (terminated or truncated) else action

        agent_idx = int(agent.split('_')[-1])
        if step_action is not None:
            self.last_actions[agent_idx] = 0.0
            self.last_actions[agent_idx, step_action] = 1.0

        self.env.step(step_action)

        if agent not in self.env.agents:
            return np.zeros(self.obs_dim, dtype=np.float32), 0.0, True, True, {}, 0

        obs_flat, team_reward, terminated, truncated, info = self.env.last(agent)
        obs = self.get_obs(agent)

        # -----------------------------------------------------------------
        # 個別評価報酬（Individual Reward）の計算開始
        # -----------------------------------------------------------------
        individual_reward = 0.0

        # エージェントの現在の実座標を取得（移動判定用）
        raw_env = self.env.unwrapped
        curr_pos = None
        try:
            agent_obj = next(a for a in raw_env.agents if a.name == agent)
            curr_pos = (agent_obj.state[1], agent_obj.state[0])  # (y, x)
        except Exception:
            pass

        # 新しい調整報酬（coop_density_reward）を受け取る
        current_min_dist, allies_count, flank_allies, shaping_reward, count_capture, coop_density_reward = self._analyze_observation(agent, obs)
        count_capture = count_capture if count_capture else 0
        team_reward += self.surround_reward * count_capture

        # 🌟 視界の中に敵がいない（索敵フェーズ）の処理
        if current_min_dist is None:
            # 視界（7x7）全体にいる味方の数をカウント
            spatial_part = obs[:self.spatial_dim]
            obs_grid = spatial_part.reshape(self.obs_range, self.obs_range, 4)
            ally_layer = obs_grid[:, :, 2]
            
            # 自分を含めた視界内の合計人数（レイヤーで要素が0超の部分の合計 + 自分自身(1)）
            allies_in_view = np.sum(ally_layer > 0) + 1

            # 移動したかどうかの判定
            prev_pos = self.prev_agent_positions.get(agent)
            has_moved = False
            if prev_pos is not None and curr_pos is not None:
                # 座標が変わっていれば移動したとみなす
                if prev_pos != curr_pos:
                    has_moved = True

            if allies_in_view == 4 and has_moved:
                # 4人チームでまとまって移動している場合はボーナス
                individual_reward += self.search_coop_move_bonus
            elif not has_moved:
                # 敵がいないのにその場にとどまり続けている（うろうろ含む）場合はペナルティ
                individual_reward += self.search_stagnation_penalty
        else:
            # 敵が視界内にいる場合は、既存の包囲シェイピング・人数最適化報酬を適用
            individual_reward += shaping_reward
            individual_reward += coop_density_reward

        # 座標の更新
        if curr_pos is not None:
            self.prev_agent_positions[agent] = curr_pos

        # 1. 距離・回り込みベースの評価
        reward_distance = 0.0
        prev_dist = self.prev_min_distances.get(agent)
        
        is_approaching = False
        if current_min_dist is not None and prev_dist is not None:
            change = prev_dist - current_min_dist
            reward_distance = change * self.distance_reward_scale
            if change > 0:
                is_approaching = True
                if flank_allies == 0:
                    reward_distance += self.flanking_bonus_scale

        self.prev_min_distances[agent] = current_min_dist
        individual_reward += reward_distance

        # 2. 協調行動（接近＋包囲網）の評価
        reward_coop = 0.0
        if current_min_dist is not None and current_min_dist <= 2:
            if allies_count >= 1:
                reward_coop += self.coop_reward_scale
            if allies_count >= 3:
                reward_coop += self.surround_reward
        individual_reward += reward_coop

        # 3. 同時アプローチ（4人が同時に接近）のチーム集計とボーナス適用
        if not hasattr(self, '_approach_registry'):
            self._approach_registry = {}
            self._cycle_bonus_given = -1

        if current_cycle not in self._approach_registry:
            self._approach_registry[current_cycle] = {}
        
        self._approach_registry[current_cycle][agent] = is_approaching

        if len(self._approach_registry[current_cycle]) >= len(self.env.agents):
            approaching_count = sum(self._approach_registry[current_cycle].values())
            if approaching_count == 4 and self._cycle_bonus_given != current_cycle:
                individual_reward += self.simultaneous_approach_bonus
                self._cycle_bonus_given = current_cycle
                self._approach_registry = {k: v for k, v in self._approach_registry.items() if k >= current_cycle}

        # 4. 衝突ペナルティの評価
        if info.get('wasted_move', False):
            individual_reward += self.collision_penalty

        # 5. 完全捕獲成功時のボーナス
        if terminated and not truncated:
            if team_reward > 0:
                time_bonus = max(0, 500 - current_cycle) * 1.0
                individual_reward += (500.0 + time_bonus)
                print(f"--- 🎉 TRUE CAPTURE SUCCESS! Agent: {agent} | Bonus: {500.0 + time_bonus} ---")

        # ハイブリッド報酬の合算
        hybrid_reward = (team_reward + individual_reward) * 0.05

        return obs, hybrid_reward, terminated, truncated, info, count_capture

    def render(self):
        render_mode = getattr(self.env, "render_mode", None)
        if render_mode is None:
            return None
        frame = self.env.render()
        if frame is None and render_mode == "human":
            try:
                import pygame
                screen = pygame.display.get_surface()
                if screen is not None:
                    img_str = pygame.image.tostring(screen, "RGB")
                    frame = np.frombuffer(img_str, dtype=np.uint8).reshape(screen.get_size()[1], screen.get_size()[0], 3)
            except Exception:
                pass
        return frame