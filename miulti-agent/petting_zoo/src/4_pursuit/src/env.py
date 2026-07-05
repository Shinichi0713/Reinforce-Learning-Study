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

        # 🌟 変更点1: 観測次元数の再定義
        # - 空間観測: 7x7x4 = 196次元
        # - 味方の行動履歴: 5行動 (ワンホット) × 8人 = 40次元
        # 合計: 236次元
        self.spatial_dim = (obs_range * obs_range * 4)
        self.action_history_dim = 5 * self.num_agents  # 5行動ワンホット × 8人分
        self.obs_dim = self.spatial_dim + self.action_history_dim
        self.state_dim = self.obs_dim * self.num_agents

        self.action_space = self.env.action_space(self.possible_agents[0])
        self.action_dim = self.action_space.n

        # ハイブリッド報酬の重みパラメータ
        self.distance_reward_scale = 0.15     # 獲物へ接近したときの報酬
        self.coop_reward_scale = 0.2          # 獲物の近くで味方と連携したときの報酬
        self.flanking_bonus_scale = 0.2       # 他に味方がいないルートから回り込んだときの報酬
        self.surround_reward = 10.0            # 3人以上で包囲網を形成したときの報酬

        # 包囲網シェイピング用のパラメータを追加
        self.soft_gather_reward_scale = 0.05    # 2マス先までに味方が集まっているとき（緩い報酬）
        self.cross_position_reward_scale = 0.25 # 上下左右のジャスト位置に配置されたとき（強めの報酬）

        # 衝突ペナルティ（進路妨害や壁への衝突による無駄な動きを抑制）
        self.collision_penalty = -0.15

        # エージェントごとの前ステップの「最も近い獲物への距離」を記録する辞書
        self.prev_min_distances = {}
        self.capture_count = 0
        self.captured_prey_ids = set()  # 捕獲済みターゲットのID（座標など）を保持

        # 🌟 変更点2: 全エージェントの直前行動を保持するグローバルバッファ (初期値は「滞在: 4」のワンホット)
        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
        self.last_actions[:, 4] = 1.0  # 4: 滞在 (NONE)

    def reset(self):
        self.env.reset()
        self.prev_min_distances = {agent: None for agent in self.possible_agents}
        # 🌟 行動履歴を初期化（滞在状態）
        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
        self.last_actions[:, 4] = 1.0

    def get_obs(self, agent):
        if agent not in self.env.agents:
            return None
        obs, _, _, _, _ = self.env.last(agent)
        if obs is None:
            return None

        # 元の環境のレイヤー (7, 7, 3) をパース
        wall_layer = obs[:, :, 0]
        ally_layer = obs[:, :, 1]
        prey_layer = obs[:, :, 2]

        # 🌟 変更点3: 味方レイヤー(Ch2)を個別ID（1.0〜8.0）で埋め直す
        # 元の ally_layer は単に 1 が立っているだけなので、内部情報からIDを特定して配置する
        id_ally_layer = np.zeros((self.obs_range, self.obs_range), dtype=np.float32)
        
        raw_env = self.env.unwrapped
        try:
            # 自分の絶対座標を取得
            my_agent_obj = next(a for a in raw_env.agents if a.name == agent)
            my_y, my_x = my_agent_obj.state[1], my_agent_obj.state[0]

            # 他の全エージェントを走査して自分のローカル視野に投影
            for other_agent in raw_env.agents:
                if other_agent.name == agent:
                    continue
                
                oy, ox = other_agent.state[1], other_agent.state[0]
                local_y = oy - my_y + self.center_idx
                local_x = ox - my_x + self.center_idx

                if 0 <= local_y < self.obs_range and 0 <= local_x < self.obs_range:
                    # 'pursuer_0' -> 1.0, 'pursuer_7' -> 8.0 のように浮動小数点ID化して格納
                    agent_id = int(other_agent.name.split('_')[-1]) + 1
                    id_ally_layer[local_y, local_x] = float(agent_id)
        except Exception:
            # 万が一のフォールバック (元のレイヤー構造をそのまま使用)
            id_ally_layer = ally_layer.astype(np.float32)

        # 1. 完全に空いている（通行可能）セルの判定 (味方ID化レイヤーを考慮)
        empty_layer = ((wall_layer == 0) & (id_ally_layer == 0) & (prey_layer == 0)).astype(np.float32)

        # 2. 4つのレイヤーを結合して (7, 7, 4) のセマンティック・テンソルを作成
        semantic_obs = np.stack([
            empty_layer,                   # Ch 0: 通行可能
            wall_layer.astype(np.float32), # Ch 1: 壁
            id_ally_layer,                 # Ch 2: 🌟 味方（個別ID仕様）
            prey_layer.astype(np.float32)  # Ch 3: 敵
        ], axis=-1)

        spatial_flat = semantic_obs.reshape(-1)  # 196次元

        # 🌟 変更点4: 全員の直前行動履歴（40次元）をフラット化して空間特徴量と結合
        action_history_flat = self.last_actions.reshape(-1)  # (8 * 5,) = 40次元
        
        # 最終的な観測（196 + 40 = 236次元）を生成
        full_obs = np.concatenate([spatial_flat, action_history_flat])
        return full_obs

    def get_global_state(self):
        """
        全員の観測（各236次元）を純粋にリスト化して結合し、
        (num_agents * 236,) の1次元の元データを作成する
        """
        obs_list = []
        for agent in self.possible_agents:
            obs_flat = self.get_obs(agent)
            if obs_flat is not None:
                obs_list.append(obs_flat)
            else:
                obs_list.append(np.zeros(self.obs_dim, dtype=np.float32))

        return np.concatenate(obs_list)

    def count_captures(self, prey_positions, ally_layer):
        count_capture = 0
        for py, px in prey_positions:
            # このターゲットの周囲（マンハッタン距離1以内）にいる味方の数を数える
            allies_around_prey = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.obs_range and 0 <= nx < self.obs_range:
                        allies_around_prey += ally_layer[ny, nx]

            # 3人以上で囲まれていて、まだ捕獲済みでないターゲットを「捕獲」とみなす
            if allies_around_prey >= 3:
                prey_id = (py, px)  # ターゲットのID（座標で代用）
                if prey_id not in self.captured_prey_ids:
                    # グローバルに「捕獲済み」として登録
                    self.captured_prey_ids.add(prey_id)
                    self.capture_count += 1
                    # print(f"--- 🎉 CAPTURE #{self.capture_count} (prey at {py},{px}) ---")
                    count_capture += 1
            return count_capture

    def _analyze_observation(self, obs_flat):
        """
        1つのエージェントの平坦化された観測(236,)から報酬計算用の情報を抽出する
        """
        if obs_flat is None:
            return None, 0, 0, 0.0

        # 🌟 変更点5: 入力次元が増えたため、先頭の196次元（空間データ）のみを切り出してデコード
        spatial_part = obs_flat[:self.spatial_dim]
        obs = spatial_part.reshape(self.obs_range, self.obs_range, 4)

        # 4チャンネル仕様のインデックス (Ch2が味方、Ch3が敵)
        ally_layer = obs[:, :, 2]
        prey_layer = obs[:, :, 3]

        prey_positions = np.argwhere(prey_layer > 0)
        # 捕獲カウント
        count_capture = self.count_captures(prey_positions, ally_layer)

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
            return None, 0, 0, 0.0, count_capture  # 視界内に獲物がいない場合

        # 2. 協調判定：自分の隣接4マスにいる味方の総数 (ID化されているため、0より大きいマスをカウント)
        allies_count = 0
        neighbors = [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]
        for ny, nx in neighbors:
            if 0 <= ny < self.obs_range and 0 <= nx < self.obs_range:
                if ally_layer[ny, nx] > 0:
                    allies_count += 1

        # 3. 獲物がいる「方向（セクター）」に他の味方が何人いるかをカウント
        py, px = closest_prey_pos
        flank_allies = 0

        if py < cy:
            flank_allies += np.sum(ally_layer[0:cy, :] > 0)
        elif py > cy:
            flank_allies += np.sum(ally_layer[cy+1:, :] > 0)

        if px < cx:
            flank_allies += np.sum(ally_layer[:, 0:cx] > 0)
        elif px > cx:
            flank_allies += np.sum(ally_layer[:, cx+1:] > 0)

        # 4. 特定のマス（最も近い敵）を基準とした包囲報酬の計算
        shaping_reward = 0.0

        # 敵の周囲（マンハッタン距離で2マス以内の全範囲を走査）
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                target_y = py + dy
                target_x = px + dx

                # 視界(7x7)の範囲内、かつ「敵のマスそのもの」は除外
                if (0 <= target_y < self.obs_range) and (0 <= target_x < self.obs_range) and (dy != 0 or dx != 0):

                    # その位置に味方が綺麗に配置されているか？ (0より大きければ味方が存在)
                    if ally_layer[target_y, target_x] > 0:
                        m_dist = abs(dy) + abs(dx)

                        # 条件A: 上下左右のジャスト位置（距離1の十字位置）の場合（強めの報酬）
                        if m_dist == 1:
                            # ※重ね合わせによる重複はID化で検知可能（必要なら厳密化できますが、ここでは存在判定）
                            shaping_reward += self.cross_position_reward_scale

                        # 条件B: 2マス先までにゆるく集まっている場合（緩い報酬）
                        elif m_dist <= 2:
                            shaping_reward += self.soft_gather_reward_scale

        return min_dist, allies_count, flank_allies, shaping_reward, count_capture

    def step(self, agent, action):
        if agent not in self.env.agents:
            return np.zeros(self.obs_dim, dtype=np.float32), 0.0, True, True, {}

        current_cycle = getattr(self.env.unwrapped, 'cycles', 0)

        _, _, terminated, truncated, _ = self.env.last(agent)
        step_action = None if (terminated or truncated) else action

        # 🌟 変更点6: エージェントが選択した行動履歴をグローバルバッファに即時ワンホット形式で書き込み
        agent_idx = int(agent.split('_')[-1])
        if step_action is not None:
            self.last_actions[agent_idx] = 0.0
            self.last_actions[agent_idx, step_action] = 1.0

        self.env.step(step_action)

        if agent not in self.env.agents:
            return np.zeros(self.obs_dim, dtype=np.float32), 0.0, True, True, {}

        obs_flat, team_reward, terminated, truncated, info = self.env.last(agent)

        # 🌟 236次元にアップデートされた新しい get_obs(agent) データを取得
        obs = self.get_obs(agent)

        # -----------------------------------------------------------------
        # 個別評価報酬（Individual Reward）の計算開始
        # -----------------------------------------------------------------
        individual_reward = 0.0

        # 観測データから距離、周囲の味方数、シェイピング報酬を抽出
        current_min_dist, allies_count, flank_allies, shaping_reward, count_capture = self._analyze_observation(obs)

        # 包囲網シェイピング報酬の適用
        individual_reward += shaping_reward

        # 1. 距離・回り込みベースの評価
        reward_distance = 0.0
        prev_dist = self.prev_min_distances.get(agent)

        if current_min_dist is not None and prev_dist is not None:
            change = prev_dist - current_min_dist
            reward_distance = change * self.distance_reward_scale
            if change > 0 and flank_allies == 0:
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

        # 3. 衝突ペナルティの評価
        if info.get('wasted_move', False):
            individual_reward += self.collision_penalty

        # 4. チーム全員の完全捕獲成功時の個別時間ボーナス
        if terminated and not truncated:
            if team_reward > 0:
                time_bonus = max(0, 500 - current_cycle) * 1.0
                individual_reward += (500.0 + time_bonus)
                print(f"--- 🎉 TRUE CAPTURE SUCCESS! Agent: {agent} | Bonus: {500.0 + time_bonus} ---")

        # -----------------------------------------------------------------
        # ハイブリッド報酬の合算（チーム共通報酬 ＋ 各自の個別評価）
        # -----------------------------------------------------------------
        hybrid_reward = team_reward + individual_reward

        # 🌟 236次元の obs（空間196 + 行動40）を返します
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