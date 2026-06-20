ご提示いただいた `PursuitWrapper` に、指定された条件（①特定のマスを囲むように緩く集まる、②上下左右に綺麗に配置される）の報酬設計を追加しましょう。

この追加報酬は、エージェントたちが「なんとなく近くに集まる（プレ・ポジショニング）」ことから始め、最終的に「綺麗な十字の包囲網（上下左右）」を完成させるための非常に強力なガイド（報酬シェイピング）になります。

---

## 🛠️ `PursuitWrapper` の修正コード

このロジックを実装するために、`_analyze_observation` の内部で「視界内に写っているすべての敵（または特定のマス）」をベースに、周囲の味方の配置をグリッド距離（マンハッタン距離）で走査する処理を追加します。

以下の変更・追加箇所を反映させてください。

### 1. `__init__` に新しい報酬スケールを追加

```python
        # （既存のパラメータの下に追加）
        # 新設：特定のマス（敵）をターゲットとした包囲網シェイピング報酬
        self.soft_gather_reward_scale = 0.05    # 2マス先までに味方が集まっているとき（緩い報酬）
        self.cross_position_reward_scale = 0.25 # 上下左右のジャスト位置に配置されたとき（強めの報酬）

```

### 2. `_analyze_observation` の修正

視界内の各「敵（特定のマス）」に対して、周囲の味方の配置（2マス以内、および上下左右）をカウントするロジックを組み込みます。

```python
    def _analyze_observation(self, obs):
        """
        1つのエージェントの観測(7, 7, 3)から報酬計算用の情報を抽出する
        """
        if obs is None:
            return None, 0, 0, 0.0

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
            return None, 0, 0, 0.0  # 視界内に獲物がいない場合

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

        # 🌟 4. 【新設】特定のマス（最も近い敵）を基準とした包囲報酬の計算
        shaping_reward = 0.0
        
        # 敵の周囲（縦横マスのオフセット）をチェック
        # マンハッタン距離で2マス以内の全範囲を走査
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                target_y = py + dy
                target_x = px + dx
                
                # 視界(7x7)の範囲内、かつ「敵のマスそのもの」は除外
                if (0 <= target_y < self.obs_range) and (0 <= target_x < self.obs_range) and (dy != 0 or dx != 0):
                    # その位置に味方がいるか（ally_layerは0または1以上）
                    if ally_layer[target_y, target_x] > 0:
                        m_dist = abs(dy) + abs(dx)
                        
                        # 条件A: 上下左右のジャスト位置（距離1の十字位置）の場合（やや強めの報酬）
                        if m_dist == 1:
                            shaping_reward += self.cross_position_reward_scale
                        # 条件B: 2マス先までにゆるく集まっている場合（緩い報酬）
                        elif m_dist <= 2:
                            shaping_reward += self.soft_gather_reward_scale

        return min_dist, allies_count, flank_allies, shaping_reward

```

### 3. `step` 内での報酬合算の修正

新しく計算した `shaping_reward` を `individual_reward` に加算します。

```python
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

        # 🛠️ 変更：戻り値に包囲シェイピング報酬（shaping_reward）を追加
        current_min_dist, allies_count, flank_allies, shaping_reward = self._analyze_observation(obs)

        # 🌟 新設：包囲網シェイピング報酬の適用
        individual_reward += shaping_reward

        # 1. 距離・回り込みベースの評価
        reward_distance = 0.0
        prev_dist = self.prev_min_distances.get(agent)
        
        # （以下、既存の処理がそのまま続きます...）

```

---

## 💡 この報酬設計の優れた効果

この追加によって、エージェントたちの学習は以下のように劇的に洗練されます。

* **段階的な包囲スキルの獲得（カリキュラム効果）**:
これまでは「ただ敵に近づく」だけだったため、1体の敵に対して味方が縦一列に並んで追従してしまうような無駄な動きが発生しがちでした。今回の修正により、「敵の近くの空間（2マス以内）」に留まるだけで微小なベース報酬が入るため、エージェントは敵の逃げ道を塞ぐように周囲に散開して待機する（ゆるい包囲）ことを覚えます。
* **「挟み込み」の誘導**:
上下左右の十字位置（距離1）にカチッとハマると高めの報酬が得られるため、エージェントたちは「空いている対面」や「上下の隙間」へと滑り込もうとする強いインセンティブが働きます。

第1回目に指摘した「捕獲成功時の莫大な報酬スケール（+500）」を `5.0` 〜 `10.0` 程度に抑えてあれば、今回の `0.05` と `0.25` というシェイピング報酬のバランスは、「ハメ技を起こさず、かつ包囲のフォーメーションを綺麗に誘導する」ためのジャストなスケール感として綺麗に機能します。