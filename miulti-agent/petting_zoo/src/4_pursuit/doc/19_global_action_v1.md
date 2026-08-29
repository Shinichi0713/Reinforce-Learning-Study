
前回[MoE均等化、順序ランダム化によるMoEが抱えるモデルの偏りを解消する方法を実装](https://yoshishinnze.hatenablog.com/entry/2026/09/26/043000)しました。
結果、MoEのルーティングの改善につながりましたが結果として、敵を捕獲した後も同じ場所に居座り続けるという問題が解消されていません。
現状から改善できないか検討していきます。

## 課題

前回記事の考察をそのまま引用しますが。

>敵が近づいたときに味方で集団で上手に囲むため、1体の敵を上手に捕獲していきます。
>ですが、以前課題と感じていた待機型に戻っています。
>スコアは前回よりも良いですが、敵が多いところにわーと集団移動するような動作がなくなってしまった感があります。

行動は状況を見て決めてもらうと良いのですが、現状は一角に待機するような動作となっています。
何とか状況を判断してもらうようにしたいというのが現時点です。

## 対応策

ここまでの対応（MoE均等化、順序ランダム化、報酬修正）はいずれも「既にある仕組みの偏りを補正する」対策でした。それでも改善しないということは、**モデルに「敵がいない場所を探しに行く」ための情報や仕組みそのものが構造的に欠けている**可能性が高いです。



## 原因の本質: エージェントは「見えている範囲」でしか判断材料がない

`get_obs()` は7x7の局所観測だけを返します。索敵報酬(`search_stagnation_penalty`など)は改善しましたが、**「罰を避けるために何かしら動く」ことは学習できても、「どちらに向かえば敵がいるか」という方向情報自体が観測に含まれていません**。局所観測が空なら、モデルにとってはどちらへ動いても「見えている情報」としては同じです。結果、学習が収束するにつれて「無難な、リスクの低い狭い範囲での徘徊」に落ち着きやすくなります。これが一番効いている可能性が高いです。

### 対策1（最優先）: 「最寄り未捕獲ターゲットへの相対方向」を観測に直接追加する

以前`order_mode="priority"`のために作った`compute_priority_scores`は、**順序決定だけに使われ、モデルの入力(観測)には渡っていません**。これをエージェントの観測特徴として直接与えます。

```python
# PursuitWrapper に追加
def get_obs(self, agent):
    ...(既存の処理はそのまま)...
    spatial_flat = semantic_obs.reshape(-1)
    action_history_flat = self.last_actions.reshape(-1)

    # 🌟 追加: グローバル情報(観測範囲外)から、最寄り未捕獲preyへの相対方向を計算
    rel_vec = self._compute_relative_direction_to_nearest_prey(agent)  # (dy, dx)を正規化した値

    full_obs = np.concatenate([spatial_flat, action_history_flat, rel_vec])
    return full_obs

def _compute_relative_direction_to_nearest_prey(self, agent) -> np.ndarray:
    raw_env = self.env.unwrapped
    try:
        evader_positions = [(e.state[1], e.state[0]) for e in raw_env.evaders]
        agent_obj = next(a for a in raw_env.agents if a.name == agent)
        ay, ax = agent_obj.state[1], agent_obj.state[0]
    except Exception:
        return np.zeros(2, dtype=np.float32)

    if not evader_positions:
        return np.zeros(2, dtype=np.float32)  # 全捕獲済み

    ty, tx = min(evader_positions, key=lambda p: abs(ay - p[0]) + abs(ax - p[1]))
    dy, dx = (ty - ay), (tx - ax)
    # マップサイズで正規化(方向のみを与え、絶対距離のスケールに依存しすぎないようにする)
    norm = max(abs(dy), abs(dx), 1)
    return np.array([dy / norm, dx / norm], dtype=np.float32)
```

`self.obs_dim` に `+2` を加え、`MATObsEncoder`側で `spatial_obs` と `action_history` に加えて、この2次元ベクトルも `feature_fuse` に連結するよう変更します。

```python
# MATObsEncoder.__init__
self.feature_fuse = nn.Linear(d_model * 2 + 2, d_model)  # 🌟 +2

# MATObsEncoder.forward
direction_feat = obs[:, -2:]  # 末尾2次元が方向ベクトル
fused = torch.cat([spatial_feature, act_emb, direction_feat], dim=-1)
return self.feature_fuse(fused)
```

**これは「視界の外の情報を使って、視界の中の行動を決める」ことを直接可能にする変更**で、局所観測とグローバル報酬シェイピングの間を埋める、最も直接的な対策です。

---

## 原因2: 「もう十分に人が集まっている」ことを判断する材料はあるが、「別の場所が手薄」ことを示す材料がない

一角に居座るのは、その場所の情報（味方の密度など）は見えていても、**他の場所に敵がいるという情報が相対的に伝わりにくい**ためとも考えられます。対策1でこれはある程度緩和されますが、さらに踏み込むなら「チーム全体でどこに散らばるべきか」を明示的に割り振る仕組みが有効です。

### 対策2: 訪問済みマップ（探索カバレッジ）による内発的報酬

「敵がいない」という消極的シグナルだけでなく、「まだ訪れていない場所に行くこと自体に価値がある」という内発的動機付け(intrinsic reward)を加えると、居座りに対してより強い力がかかります。

```python
# PursuitWrapper に追加
def reset(self):
    ...(既存)...
    self.visit_counts = {}  # {(y, x): 訪問回数}

def _get_exploration_bonus(self, curr_pos) -> float:
    count = self.visit_counts.get(curr_pos, 0)
    self.visit_counts[curr_pos] = count + 1
    # 訪問回数が少ないマスほど高いボーナス(count-based exploration)
    return 1.0 / np.sqrt(count + 1)
```

`step()`の索敵フェーズ内で、このボーナスを`search_approach_bonus`に加えて併用します。

```python
if not has_moved:
    individual_reward += self.search_stagnation_penalty
else:
    exploration_bonus = self._get_exploration_bonus(curr_pos) * 0.02  # 係数は調整
    individual_reward += exploration_bonus
    ...(既存のsearch_approach_bonus判定)...
```

これは「敵に近づいたか」に関係なく、**単純に「新しい場所に行った」ことを評価する**ので、たとえ敵の手がかりが全くない状況でも、居座りから抜け出す動機になります。対策1（方向情報）と組み合わせると、「手がかりがあればそちらへ、なければ未探索エリアへ」という自然な振る舞いに近づけやすくなります。

---

## 原因3: モデル構造として「チーム内での役割分担」を強制する仕組みがない

対策1・2は「個々のエージェントに情報を与える」アプローチですが、**「8体全員が同じ情報に基づいて同じように"近くに集まる"最適化をしてしまう」リスク**は残ります。全員が同じ観測(相対方向)を見て、全員が同じ最寄りターゲットを目指せば、結局同じ場所に集まってしまいかねません。

### 対策3: エージェントごとに異なる担当エリア/ターゲットを明示的に割り当てる

前回の`priority_agent_order`をベースに、**割り当て(assignment)そのものをハンガリアン法などで最適化し、各エージェントに「自分が担当するターゲット」を固定して観測に含める**ようにすると、この重複問題を根本的に避けられます。

```python
from scipy.optimize import linear_sum_assignment

def compute_agent_target_assignment(self) -> dict:
    """
    各pursuerに、重複しないよう1体ずつ異なる未捕獲preyを割り当てる(距離コストの最小化)。
    prey数 < agent数の場合、余ったagentは「最も近いprey」に複数人割り当てる。
    """
    raw_env = self.env.unwrapped
    evader_positions = [(e.state[1], e.state[0]) for e in raw_env.evaders]
    if not evader_positions:
        return {agent: None for agent in self.possible_agents}

    agent_positions = []
    valid_agents = []
    for agent in self.possible_agents:
        if agent not in self.env.agents:
            continue
        try:
            obj = next(a for a in raw_env.agents if a.name == agent)
            agent_positions.append((obj.state[1], obj.state[0]))
            valid_agents.append(agent)
        except Exception:
            continue

    cost_matrix = np.array([
        [abs(ay - py) + abs(ax - px) for (py, px) in evader_positions]
        for (ay, ax) in agent_positions
    ])

    # prey数がagent数より少ない場合はタイル(繰り返し)してコスト行列を拡張
    n_agents, n_prey = cost_matrix.shape
    if n_prey < n_agents:
        reps = int(np.ceil(n_agents / n_prey))
        cost_matrix = np.tile(cost_matrix, (1, reps))[:, :n_agents]

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    assignment = {}
    for r, c in zip(row_idx, col_idx):
        prey_idx = c % n_prey
        assignment[valid_agents[r]] = evader_positions[prey_idx]

    return assignment
```

この割り当てを使って、対策1の「最寄りターゲットへの相対方向」を**「最寄り」ではなく「自分に割り当てられたターゲット」への方向**に差し替えると、**エージェントが自然に散らばって別々のターゲットを追う**ようになります。同じターゲットに全員が向かって過密になる、という現在の症状に対して、より直接的な効果が期待できます。

---

## 優先順位のまとめ

| 対策 | 効くはずの症状 | 実装コスト |
|---|---|---|
| 1. 方向情報を観測に追加 | 「そもそもどちらに探索すべきか分からない」 | 低 |
| 2. 訪問カバレッジによる内発的報酬 | 「手がかりがなくても居座らず動く」 | 低〜中 |
| 3. ハンガリアン法による明示的な担当割当 | 「一角に全員集中してしまう」重複問題 | 中〜高 |

**まず対策1（方向情報の追加）を試すことを強くお勧めします**。これまでの報酬シェイピングだけでは「動機」は与えられても「方向」の手がかりがなかったので、恐らく最も本質的なギャップです。それでも一角への集中が残るようなら、対策3（明示的な担当割当）に進むのが筋道として自然だと思います。対策2はどちらとも独立して併用可能なので、余力があれば追加してみてください。
