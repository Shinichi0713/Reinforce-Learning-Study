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

        self.obs_dim = (obs_range * obs_range * 3)
        self.state_dim = self.obs_dim * self.num_agents

        self.action_space = self.env.action_space(self.possible_agents[0])
        self.action_dim = self.action_space.n

        # ハイブリッド報酬の重みパラメータ
        self.distance_reward_scale = 0.15     # 獲物へ接近したときの報酬
        self.coop_reward_scale = 0.2         # 獲物の近くで味方と連携したときの報酬
        self.flanking_bonus_scale = 0.2      # 他に味方がいないルートから回り込んだときの報酬
        self.surround_reward = 2.0           # 3人以上で包囲網を形成したときの報酬

        # 衝突ペナルティ（進路妨害や壁への衝突による無駄な動きを抑制）
        self.collision_penalty = -0.1

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

        # 3. 獲物がいる「方向（セクター）」に他の味方が何人いるかをカウント
        py, px = closest_prey_pos
        flank_allies = 0

        if py < cy:
            flank_allies += np.sum(ally_layer[0:cy, :])
        elif py > cy:
            flank_allies += np.sum(ally_layer[cy+1:, :])

        if px < cx:
            flank_allies += np.sum(ally_layer[:, 0:cx])
        elif px > cx:
            flank_allies += np.sum(ally_layer[:, cx+1:])

        return min_dist, allies_count, flank_allies

    def step(self, agent, action):
        if agent not in self.env.agents:
            return 0.0, True, True, {}

        current_cycle = getattr(self.env.unwrapped, 'cycles', 0)

        _, _, terminated, truncated, _ = self.env.last(agent)
        step_action = None if (terminated or truncated) else action

        self.env.step(step_action)

        if agent not in self.env.agents:
            return 0.0, True, True, {}

        obs, team_reward, terminated, truncated, info = self.env.last(agent)

        # -----------------------------------------------------------------
        # 🌟 個別評価報酬（Individual Reward）の計算開始
        # -----------------------------------------------------------------
        individual_reward = 0.0

        # 観測データから距離、周囲の味方数を抽出
        current_min_dist, allies_count, flank_allies = self._analyze_observation(obs)

        # 1. 距離・回り込みベースの評価
        reward_distance = 0.0
        prev_dist = self.prev_min_distances.get(agent)

        if current_min_dist is not None and prev_dist is not None:
            change = prev_dist - current_min_dist
            reward_distance = change * self.distance_reward_scale
            # 獲物に向かって進んでおり、かつそのルートにまだ味方がいない（孤立して回り込んでいる）場合ボーナス
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

        # 3. 衝突ペナルティの評価（無駄な移動の抑制）
        # Pursuit環境では移動が失敗した際（衝突等）に info['wasted_move'] が True になります
        if info.get('wasted_move', False):
            individual_reward += self.collision_penalty

        # 4. チーム全員の完全捕獲成功時の個別時間ボーナス
        if terminated and not truncated:
            if team_reward > 0:
                time_bonus = max(0, 500 - current_cycle) * 1.0
                individual_reward += (500.0 + time_bonus)
                print(f"--- 🎉 TRUE CAPTURE SUCCESS! Agent: {agent} | Bonus: {500.0 + time_bonus} ---")

        # -----------------------------------------------------------------
        # 🌟 ハイブリッド報酬の合算（チーム共通報酬 ＋ 各自の個別評価）
        # -----------------------------------------------------------------
        hybrid_reward = team_reward + individual_reward

        return hybrid_reward, terminated, truncated, info

    def render(self):
        """
        環境の現在の画面をキャプチャして返します。
        推論用スクリプトで frames.append() して gif に保存するために使用します。
        """
        # 1. 内部環境（pettingzooのenv）の現在の render_mode を取得
        render_mode = getattr(self.env, "render_mode", None)
        
        if render_mode is None:
            return None
            
        # 2. mode が "human" もしくは "rgb_array" の場合、env.render() を呼び出す
        # pursuit_v4 の仕様上、render() は直接画像（numpy array）を返すか、
        # もしくは内部のスクリーンバッファから画像を取得する必要があります。
        frame = self.env.render()
        
        # 3. もし render_mode="human" で、かつ render() が None を返す（画面描画のみ行う）場合、
        # pygame のサーフェスから直接ピクセル配列をキャプチャします。
        if frame is None and render_mode == "human":
            try:
                import pygame
                # 現在表示されている pygame 画面を文字列（RGB）として取得し、numpy 配列に変換
                screen = pygame.display.get_surface()
                if screen is not None:
                    # ピクセルデータを RGB 形式のバイト文字列で取得
                    img_str = pygame.image.tostring(screen, "RGB")
                    # numpy 配列 [横, 縦, 3] に変換してから、一般的な画像形式 [縦, 横, 3] に変形
                    frame = np.frombuffer(img_str, dtype=np.uint8).reshape(screen.get_size()[1], screen.get_size()[0], 3)
            except Exception as e:
                # pygame が入っていない、もしくはインポートエラーなどの保険
                pass
                
        return frame