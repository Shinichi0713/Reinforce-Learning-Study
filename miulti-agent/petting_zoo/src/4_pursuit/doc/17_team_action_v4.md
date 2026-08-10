
本日はチーム連携の強化学習を行って捕獲を行っていくPursuitのトライアルについてです。
前回改修で以下を課題として、改修を行いました。
- Cross-Attentionのゲートが「クエリ（自分自身）」だけから計算されている
- チーム全体の同時クリッピングが、遅れているエージェントの学習を妨げる可能性

ですが劇的before & afterとはなりませんでした。
現状でチーム連携は出来ているのですが、一角を捕獲し尽すと停滞してしまう傾向がありました。

本日テーマ：
>味方集結して、捕獲しつくした後にすぐに切り替え出来るようにする

## 課題の整理

開始直後には問題ありません。味方を目指して一角に集結する動作を行います。
そして、その後のチーム連携の動作も問題ありません。

課題となっているのは、一角の敵をとりつくした後も、チームで一角に居座り続けるという挙動を行っています。

結果としてより高得点を目指すことが難しくなっています。

一角をとりつくした後は次の行動＝チームとして敵を探索しに行くという行動を推奨するようにする必要があります。

## 課題の原因と対策
課題としている理由についてコードと見比べっこして確認してみました。

### 原因1: 索敵フェーズの報酬設計が「留まる」ことを積極的には罰していない

`PursuitWrapper.step()` の索敵フェーズ（`current_min_dist is None`、つまり視界内に敵がいない）の報酬ロジックを見てみます。

```python
if current_min_dist is None:
    allies_in_view = np.sum(ally_layer > 0) + 1
    ...
    if allies_in_view == 4 and has_moved:
        individual_reward += self.search_coop_move_bonus       # +0.05
    elif not has_moved:
        individual_reward += self.search_stagnation_penalty    # -0.1
```

一見「留まったら罰則がある」ように見えますが、よく見ると**条件分岐が `allies_in_view == 4 and has_moved` のケース以外は全部 `elif not has_moved` に落ちるわけではありません**。具体的には：

- `allies_in_view == 4` かつ **動いていない** → どちらの分岐にも入らず、`individual_reward += 0`（何も加算されない）
- `allies_in_view != 4` かつ 動いていない → `search_stagnation_penalty`(-0.1) が適用される
- `allies_in_view != 4` かつ 動いている → 何も加算されない（ボーナスなし、罰則なし）

つまり、**「4人ちょうど集まった状態で止まっている」場合だけ、唯一ペナルティが免除される**という抜け穴があります。一角を捕獲し終えた直後、生き残った4体（あるいはそれ以上）がちょうど集まった状態で近くにいると、この条件に該当してペナルティを受けずに停止し続けられてしまいます。

さらに、`allies_in_view` は「視界7x7以内」に限定されているため、実際には索敵行動が必要な広いマップでも、**近くの味方の頭数さえ数が合えば「留まる」ことが咎められない**設計になっています。

__対策__

- 停滞ペナルティを `allies_in_view` の値に関わらず一律で適用する（「4人ちょうど揃っている場合の免除」をなくす）
- あるいは、免除条件を「4人揃っていて、かつ**そのうち少なくとも1体は動いている**」のようなチーム単位の判定にする（個々のエージェントの `has_moved` だけでなくチーム全体で見る）

```python
if current_min_dist is None:
    allies_in_view = np.sum(ally_layer > 0) + 1
    prev_pos = self.prev_agent_positions.get(agent)
    has_moved = prev_pos is not None and curr_pos is not None and prev_pos != curr_pos

    if has_moved:
        if allies_in_view == 4:
            individual_reward += self.search_coop_move_bonus
        # 動いていれば人数に関わらず停滞ペナルティは受けない(現状維持)
    else:
        # 🌟 修正: 4人揃っていても動いていなければペナルティを免除しない
        individual_reward += self.search_stagnation_penalty
```

### 原因2: 「その場にとどまる」ことが、学習された方策としてローカルな最適解になっている（探索不足）

これは②③で対処してきた「連携の構造」とは別の、**MoEのcollapse**（前回議論して未実装のまま残っている問題）とも関係が深いです。

`search_stagnation_penalty=-0.1` に対して、動いて索敵し次のターゲットに辿り着くまでの過程は、`distance_reward_scale=0.001`（かなり小さい）でしか報われません。つまり：

- **留まることの罰**: -0.1/step（明確で即時）
- **正しく索敵に動くことの報酬**: ターゲットが見つかるまでは基本的に何も得られない（見つかってようやく `distance_reward_scale` による小さな報酬が始まる）

このバランスだと、探索空間が広い（マップが大きい、残りpreyが少ない）状況ほど、「留まって-0.1を受け続ける」方が「見えないゴールを求めて動き回る」より学習上"損切りしやすい"局所解になりがちです。特にエントロピー係数 `entropy_coef=0.01` が小さいと、学習が進むにつれ探索的な行動が失われ、この局所解に収束しやすくなります。

__対策__

- `search_coop_move_bonus` を明確に「動いて索敵している」ことに対して継続的に与える設計にする（現状は`allies_in_view==4`の場合のみで発生条件が狭すぎます）
- 索敵時の移動そのものに小さな正の報酬を与える（`has_moved`なら常に微小ボーナス、動かなければ罰、というシンプルな設計に一度戻して検証する）
- `entropy_coef` を一時的に上げて（例: `0.01 → 0.03`）、停滞という局所解からの脱出を促す


## 実装
以下の2点を実装します。

1. **停滞ペナルティの抜け穴を撤廃**: 「4人ちょうど視界内にいれば免除」という条件をなくし、周囲に敵がいない時に動かなければ常に罰する
2. **索敵中のチーム単位の接近報酬**: 局所観測内には敵が見えなくても、環境の実座標（`raw_env`）を使って「最寄りの未捕獲preyまでの距離」を計算し、それが縮まった上で**チームの一定数以上が同時に接近**した場合にボーナスを与える

局所観測だけでは「敵の方向」が分からないため、報酬計算にはグローバル情報（`raw_env`）を使います。これは観測（エージェントに見せる入力）を変えるわけではなく、あくまで報酬シェーピングのための特権情報利用なので、CTDE（中央集権的学習）の枠組みでは一般的な手法です。

### 変更箇所1. `__init__` に状態追跡用の変数とパラメータを追加

```python
def __init__(self, render_mode=None, max_cycles=500, obs_range=7):
    ...(既存のコードはそのまま)...

    # 🌟 追加: 索敵フェーズ用の追加パラメータ
    self.search_stagnation_penalty = -0.1   # 既存。仕様は変更せず流用（免除条件だけ撤廃）
    self.search_approach_bonus = 0.05       # 🌟 追加: 個別に、未捕獲preyへグローバル距離が縮まったときの報酬
    self.search_team_approach_bonus = 0.2   # 🌟 追加: チーム(規定人数以上)が同時に接近したときの追加ボーナス
    self.search_team_approach_threshold = 3 # 🌟 追加: 「チームで接近」とみなす最低人数

    # 🌟 追加: 索敵フェーズでのグローバル距離追跡用
    self.prev_global_min_dist = {}
    self._search_approach_registry = {}
    self._search_team_bonus_given_cycle = -1
```

### 変更箇所2. `reset()` に初期化を追加

```python
def reset(self):
    self.env.reset()
    self.prev_min_distances = {agent: None for agent in self.possible_agents}
    self.prev_agent_positions = {agent: None for agent in self.possible_agents}
    self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
    self.last_actions[:, 4] = 1.0
    self.capture_count = 0
    self.captured_prey_ids = set()

    # 🌟 追加
    self.prev_global_min_dist = {agent: None for agent in self.possible_agents}
    self._search_approach_registry = {}
    self._search_team_bonus_given_cycle = -1
```

### 変更箇所3. グローバル距離を計算するヘルパーを追加

前回実装した `compute_priority_scores` とほぼ同じロジックですが、単一エージェント用に切り出します。既存の `compute_priority_scores` があるなら、内部でこちらを呼び出す形に統一しても構いません。

```python
def _compute_global_min_dist_to_evader(self, agent) -> float | None:
    """
    局所観測(7x7)の視界に関わらず、環境の実座標を使って
    エージェントから最寄りの未捕獲preyまでのマンハッタン距離を計算する。
    報酬シェーピング専用(観測には使わない特権情報)。
    未捕獲preyが存在しない、または座標取得に失敗した場合はNoneを返す。
    """
    raw_env = self.env.unwrapped

    try:
        evader_positions = [(e.state[1], e.state[0]) for e in raw_env.evaders]
    except Exception:
        return None

    if not evader_positions:
        return None

    try:
        agent_obj = next(a for a in raw_env.agents if a.name == agent)
        ay, ax = agent_obj.state[1], agent_obj.state[0]
    except Exception:
        return None

    return min(abs(ay - py) + abs(ax - px) for py, px in evader_positions)
```

### 変更箇所4. `step()` の索敵フェーズ部分を修正

該当箇所（`if current_min_dist is None:` のブロック）を以下に差し替えてください。

```python
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

        individual_reward = 0.0

        raw_env = self.env.unwrapped
        curr_pos = None
        try:
            agent_obj = next(a for a in raw_env.agents if a.name == agent)
            curr_pos = (agent_obj.state[1], agent_obj.state[0])
        except Exception:
            pass

        current_min_dist, allies_count, flank_allies, shaping_reward, count_capture, coop_density_reward = self._analyze_observation(agent, obs)
        count_capture = count_capture if count_capture else 0
        team_reward += self.surround_reward * count_capture

        # -----------------------------------------------------------------
        # 🌟 索敵フェーズ(視界内に敵がいない)の処理 — ここを修正
        # -----------------------------------------------------------------
        if current_min_dist is None:
            prev_pos = self.prev_agent_positions.get(agent)
            has_moved = False
            if prev_pos is not None and curr_pos is not None:
                if prev_pos != curr_pos:
                    has_moved = True

            # 🌟 修正1: 停滞ペナルティは人数条件に関わらず一律で適用する
            #    (「4人揃っていれば免除」という抜け穴を撤廃)
            if not has_moved:
                individual_reward += self.search_stagnation_penalty
                is_search_approaching = False
            else:
                # 🌟 修正2: グローバル座標を使い、未捕獲preyへの距離が縮まったかを評価
                current_global_dist = self._compute_global_min_dist_to_evader(agent)
                prev_global_dist = self.prev_global_min_dist.get(agent)

                is_search_approaching = False
                if current_global_dist is not None and prev_global_dist is not None:
                    if current_global_dist < prev_global_dist:
                        individual_reward += self.search_approach_bonus
                        is_search_approaching = True

                self.prev_global_min_dist[agent] = current_global_dist

            # 🌟 修正3: チーム単位で同時に接近しているかを集計し、
            #    規定人数以上が同時接近していれば追加ボーナス
            if current_cycle not in self._search_approach_registry:
                self._search_approach_registry[current_cycle] = {}
            self._search_approach_registry[current_cycle][agent] = is_search_approaching

            if len(self._search_approach_registry[current_cycle]) >= len(self.env.agents):
                approaching_count = sum(self._search_approach_registry[current_cycle].values())
                if (approaching_count >= self.search_team_approach_threshold
                        and self._search_team_bonus_given_cycle != current_cycle):
                    individual_reward += self.search_team_approach_bonus
                    self._search_team_bonus_given_cycle = current_cycle
                    self._search_approach_registry = {
                        k: v for k, v in self._search_approach_registry.items() if k >= current_cycle
                    }
        else:
            # 敵が視界内にいる場合は、既存の包囲シェイピング・人数最適化報酬を適用
            individual_reward += shaping_reward
            individual_reward += coop_density_reward
            # 🌟 視界内に敵が現れたら、索敵フェーズ用のグローバル距離追跡はリセットしておく
            self.prev_global_min_dist[agent] = None

        # 座標の更新
        if curr_pos is not None:
            self.prev_agent_positions[agent] = curr_pos

        # 1. 距離・回り込みベースの評価(視界内に敵がいる場合のみ意味を持つ既存ロジック)
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

        # 2. 協調行動(接近＋包囲網)の評価
        reward_coop = 0.0
        if current_min_dist is not None and current_min_dist <= 2:
            if allies_count >= 1:
                reward_coop += self.coop_reward_scale
            if allies_count >= 3:
                reward_coop += self.surround_reward
        individual_reward += reward_coop

        # 3. 同時アプローチ(視界内に敵が見えている状態での4人同時接近)のチーム集計とボーナス適用
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

        hybrid_reward = (team_reward + individual_reward) * 0.05

        return obs, hybrid_reward, terminated, truncated, info, count_capture
```

### 主な変更点の解説

__修正1: 停滞ペナルティの一律適用__

旧コードは `elif not has_moved:` という分岐で、「`allies_in_view == 4 and has_moved`」の条件に合致しない場合の**一部だけ**しかペナルティが発生しませんでした（`allies_in_view==4` かつ動いていない場合は両方の条件に該当せず何も起きなかった）。新コードでは `if not has_moved:` を最初の分岐にして、**視界内に敵がいない状態で動いていなければ、味方の人数に関係なく常にペナルティ**が入るようにしました。

__修正2: グローバル距離を使った個別接近報酬__

`_compute_global_min_dist_to_evader` で、局所観測に映っていない敵も含めた「実際に一番近い未捕獲prey」までの距離を計算し、前ステップと比較して縮まっていれば `search_approach_bonus` を与えます。これにより、**視界の外にいる敵に向かって正しい方向に歩いている場合にだけ報酬**が発生します（ランダムに動いただけでは平均的には縮まらないので、報われません）。

__修正3: チーム単位の同時接近ボーナス__

既存の「視界内に敵がいる場合の `simultaneous_approach_bonus`」と同じレジストリパターンを、索敵フェーズ専用に複製しました（`_search_approach_registry`）。**同じサイクル内で `search_team_approach_threshold`（デフォルト3）人以上が同時にグローバル距離を縮めていれば**、それぞれに `search_team_approach_bonus` を追加で与えます。これが「チーム単位で敵に向かって移動する」ことへの直接的な報酬になります。


