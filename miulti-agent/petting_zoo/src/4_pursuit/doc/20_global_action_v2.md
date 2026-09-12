
前回[MoE均等化、順序ランダム化によるMoEが抱えるモデルの偏りを解消する方法を実装](https://yoshishinnze.hatenablog.com/entry/2026/09/26/043000)しました。
結果、MoEのルーティングの改善につながりましたが結果として、敵を捕獲した後も同じ場所に居座り続けるという問題が解消されていません。
現状から改善できないか検討していきます。

## 課題

前々回記事の考察を引用しますが。

>敵が近づいたときに味方で集団で上手に囲むため、1体の敵を上手に捕獲していきます。
>ですが、以前課題と感じていた待機型に戻っています。
>スコアは前回よりも良いですが、敵が多いところにわーと集団移動するような動作がなくなってしまった感があります。

この一角にエージェントが居座るという問題に対して改善することを考えています。

## 対策の骨組

今回対策はエージェントが狙うターゲットの割り当てをハンガリアンアルゴリズムで行うというもので対策していきます。


### 対応する問題: 「一角の敵が少なくなった後もとどまろうとする味方がいる」
一角に居座るという症状の背景には、これまでの調査で以下のような要因が積み重なっていることが原因だと想定しています。

1. **MoEのcollapse**（一部のexpertしか使われず、状況に応じた行動の切り替えが学習されにくい）→ 負荷分散損失で対処
2. **デコード順序の固定**（特定のエージェントが常に受動的な役割になる）→ ランダム順序化で対処
3. **索敵報酬の抜け穴**（4人揃っていれば停滞が免除される等）→ 報酬ロジックの修正で対処
4. **視界の外に手がかりがない**（局所観測だけでは、どちらに敵がいるか分からない）→ 対策1（方向情報の追加）で対処

しかし対策1を入れても、**「全員が同じ最寄りターゲットに向かう」という判断基準そのものは変わっていません**。敵が少なくなった終盤（例えば残り2体、味方8体）の状況では、

- 8体中の多くが「一番近い敵はこっちだ」と同じ1体を指し示してしまい
- 結果としてそのターゲットに集中し、**もう1体の敵の方には誰も向かわない**

という構造的な偏りが残ります。これは「居座り」というより正確には「**目が向いていない敵が放置される**」現象で、見た目には「捕獲が進まず、そこにとどまっているように見える」症状として現れていたと考えられます。ハンガリアン法は、この「全員が同じ判断をしてしまう」という構造的な原因そのものを解消するための対策です。

### 対応していない問題: 「捕獲でまごまごする（何度も取り逃がす）」

こちらの症状については、**原因は全く別の場所**にあります。

> 現在のモデルは、各タイムステップの観測を独立したスナップショットとしてTransformerに通しています。獲物がどの方向に動いているか（速度・進行方向）の情報が一切ありません。

ハンガリアン法は「誰が、どのターゲットを追うべきか」という**割り当ての最適化**であり、「追っている最中に、逃げる敵にどう先回りするか」という**追跡そのものの精度**には一切関与しません。したがって、この症状に対しては、以前提案した以下のような対策が別途必要です。

- 獲物の動き（速度・進行方向）を観測チャンネルに追加する
- 時間方向の記憶（GRUなど）を導入し、複数ステップの動きのパターンを学習できるようにする
- 未来位置予測を補助タスクとして学習させる

### なぜハンガリアンを使うか

一言でいうと「**8体全員が同じ判断基準（最寄りのターゲット）で動くと、全員が同じ場所に集まってしまう**」という問題を、数学的に解決するために導入しました。


__これまでの対策1（対策1：方向情報の追加）で何が足りなかったか__

対策1では、各エージェントに「一番近い未捕獲の敵はどっちの方向か」という情報を渡しました。しかしこれには構造的な欠陥があります。

**8体のエージェントが全員、同じ計算式（`min(距離)`）を使っている**ため、もし敵が2体しか残っていない状況で、8体のうち5体にとって「一番近い敵」が偶然同じ1体だったとします。すると、その5体は全員「その敵の方向」を指し示され、**全員がそこに向かって集まってしまいます**。もう1体の敵は誰も見ていない、という状態が起きえます。これはまさに「一角に居座る/敵が少なくなっても捜索していかない」という報告いただいた症状と一致する構造的な原因です。

つまり対策1は「どちらに動けばいいか」というヒントは与えましたが、**「チームとして手分けする」という発想は一切入っていなかった**のです。

__ハンガリアン法が解決すること__

ハンガリアン法（割当問題の最適解を求めるアルゴリズム）を使うことで、

> 「8体のエージェント」と「残っている敵」の組み合わせを、**全体の移動コスト（距離の合計）が最小になるように、重複なく割り振る**

という計算を毎ステップ行うようにしました。

具体例で言うと：
- エージェントA、Bが両方とも「敵Xが一番近い」と感じていても
- 実は「AはXに、BはYに向かった方が、チーム全体としての移動コストの合計は少なくて済む」という組み合わせがあれば
- ハンガリアン法はそちらを選びます

これにより、**「全員が目先の最寄りに殺到する」のではなく、「チームとして手分けして効率よく散らばる」という動きが、計算上、最初から保証される**ようになります。

__なぜこれが「モデルの学習に任せる」より優れているか__

本来、「誰がどの敵を追うべきか」という役割分担は、強化学習のモデル自身に学習させたい理想的な能力です。しかし、これまで議論してきた通り：

- Transformerのattentionによる暗黙的な学習には限界があり
- MoEのcollapseや、デコード順序の固定など、様々な要因でこの「暗黙の役割分担」がうまく学習されていない状態が続いていました

そこで今回は、**「役割分担」という組合せ最適化の部分だけは、学習に頼らず数学的に確実に解いてしまい、エージェント（モデル）には『割り当てられた1体を、具体的にどう追い込むか』という、より簡単で学習しやすい問題だけを解かせる**、という設計に切り替えました。

これは前々回お伝えした「階層型設計」の考え方そのものです。

> 複数エージェントが複数ターゲットに対して「誰がどこを追うか」を決めるのは、本質的に組合せ最適化問題（assignment problem）です。これをEnd-to-EndのRLだけに任せると、特にMARLでは学習が難しく、非効率な割当（重複追跡、放置ターゲット）に陥りやすい

**「割り当てそのものはアルゴリズムに任せ、RLは低レベルの追跡動作だけに集中させる」** という役割分担を、今回のコードで実際に組み込んだ形になります。

__期待される効果と、期待していないこと__

**期待している効果**:
- 敵が複数残っている状況で、エージェントが自然に分散して手分けするようになる
- 「一角に居座る」「敵が少なくなっても他を探さない」という、まさに今回問題視されていた症状が、構造的に起きにくくなる

**注意点として、これは万能ではありません**:
- ハンガリアン法はあくまで「今この瞬間の距離」だけを見て割り振るので、**敵がどちらに逃げるかという動きの予測(以前議論した"まごつき"の問題)までは解決しません**
- 敵の動きによって割り当てが頻繁に入れ替わる可能性がある（実装時に補足した「チャタリング」の懸念）ため、そこは追加のチューニングが必要になる場合があります
- あくまで「pursuer側の座標」と「敵の座標」という、モデルの観測には含まれない特権的なグローバル情報を使って計算しているので、**実際にモデルが学習で獲得すべき「連携」の一部を、外側から補助している**という性質のものです。将来的にモデル自身がこうした役割分担を学習できるようになることが理想ですが、現状はそこまで到達していないための実務的な対応、という位置付けです


## 実装
以下、対策1（相対方向の観測追加）を土台に、**「最寄り」ではなく「自分に割り当てられたターゲット」への方向を使う**形に差し替えます。ハンガリアン法（`scipy.optimize.linear_sum_assignment`）で、pursuer-preyの距離コストを最小化する組み合わせを毎ステップ計算します。

## 1. `PursuitWrapper.__init__` に依存関係を追加

```python
import numpy as np
from pettingzoo.sisl import pursuit_v4
from scipy.optimize import linear_sum_assignment  # 🌟 追加
```

## 2. `reset()` に割り当てキャッシュ用の変数を追加

```python
def reset(self):
    self.env.reset()
    self.prev_min_distances = {agent: None for agent in self.possible_agents}
    self.prev_agent_positions = {agent: None for agent in self.possible_agents}
    self.last_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
    self.last_actions[:, 4] = 1.0
    self.capture_count = 0
    self.captured_prey_ids = set()

    self.prev_global_min_dist = {agent: None for agent in self.possible_agents}
    self._search_approach_registry = {}
    self._search_team_bonus_given_cycle = -1

    # 🌟 追加: エージェントごとの担当ターゲット割り当てキャッシュ
    #    毎ステップ計算すると同じ状態に対して重複計算になるため、
    #    サイクル単位でキャッシュする
    self._assignment_cache = {}
    self._assignment_cache_cycle = -1
```

## 3. ハンガリアン法による割り当て計算メソッドを追加

```python
def _compute_agent_target_assignment(self) -> dict:
    """
    各pursuerに、重複しないよう1体ずつ異なる未捕獲preyを割り当てる
    (マンハッタン距離コストの総和を最小化、ハンガリアン法)。

    - 未捕獲preyがpursuer数より少ない場合は、preyをタイル(繰り返し)して
      余ったpursuerも(重複を許して)最も割の良いpreyに割り当てる。
    - 未捕獲preyが0体の場合、全pursuerに None を割り当てる。
    - 座標取得に失敗したpursuerには None を割り当てる。

    戻り値: {agent_name: (target_y, target_x) or None}
    """
    raw_env = self.env.unwrapped

    try:
        evader_positions = [(e.state[1], e.state[0]) for e in raw_env.evaders]
    except Exception:
        return {agent: None for agent in self.possible_agents}

    if not evader_positions:
        return {agent: None for agent in self.possible_agents}

    # 生存しているpursuerの座標を集める
    valid_agents = []
    agent_positions = []
    for agent in self.possible_agents:
        if agent not in self.env.agents:
            continue
        try:
            obj = next(a for a in raw_env.agents if a.name == agent)
            agent_positions.append((obj.state[1], obj.state[0]))
            valid_agents.append(agent)
        except Exception:
            continue

    assignment = {agent: None for agent in self.possible_agents}
    if not valid_agents:
        return assignment

    n_agents = len(valid_agents)
    n_prey = len(evader_positions)

    # コスト行列: (n_agents, n_prey)のマンハッタン距離
    cost_matrix = np.array([
        [abs(ay - py) + abs(ax - px) for (py, px) in evader_positions]
        for (ay, ax) in agent_positions
    ], dtype=np.float64)

    # 🌟 pursuer数 > prey数の場合、コスト行列の列をタイルして
    #    全pursuerに割り当てが行き渡るようにする
    #    (同じpreyに複数pursuerが割り当てられることを許容する)
    if n_prey < n_agents:
        reps = int(np.ceil(n_agents / n_prey))
        tiled_cost = np.tile(cost_matrix, (1, reps))[:, :n_agents]
        # linear_sum_assignmentは正方 or 長方行列どちらも扱えるが、
        # ここでは「列数 >= 行数」にしておくと解釈がシンプルになる
        row_idx, col_idx = linear_sum_assignment(tiled_cost)
        for r, c in zip(row_idx, col_idx):
            prey_idx = c % n_prey  # タイルした分を元のprey индексに戻す
            assignment[valid_agents[r]] = evader_positions[prey_idx]
    else:
        # pursuer数 <= prey数の通常ケース: 1対1の最適割り当て
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_idx, col_idx):
            assignment[valid_agents[r]] = evader_positions[c]

    return assignment

def _get_agent_assignment(self, agent):
    """
    現在のサイクルの割り当て結果をキャッシュから取得する。
    (毎ステップ・毎エージェントで同じ計算を繰り返さないための最適化)
    """
    current_cycle = getattr(self.env.unwrapped, 'cycles', 0)
    if self._assignment_cache_cycle != current_cycle:
        self._assignment_cache = self._compute_agent_target_assignment()
        self._assignment_cache_cycle = current_cycle
    return self._assignment_cache.get(agent)
```

## 4. 方向計算メソッドを「最寄り」から「割り当て先」に差し替え

前回実装した `_compute_relative_direction_to_nearest_prey` を置き換えます（メソッド名も実態に合わせて変更します）。

```python
def _compute_relative_direction_to_assigned_prey(self, agent) -> np.ndarray:
    """
    自分に割り当てられたターゲット(ハンガリアン法による)への相対方向を返す。
    割り当てがない(全捕獲済み、または座標取得失敗)場合はゼロベクトル。
    """
    raw_env = self.env.unwrapped
    try:
        agent_obj = next(a for a in raw_env.agents if a.name == agent)
        ay, ax = agent_obj.state[1], agent_obj.state[0]
    except Exception:
        return np.zeros(2, dtype=np.float32)

    target = self._get_agent_assignment(agent)
    if target is None:
        return np.zeros(2, dtype=np.float32)

    ty, tx = target
    dy, dx = (ty - ay), (tx - ax)
    norm = max(abs(dy), abs(dx), 1)
    return np.array([dy / norm, dx / norm], dtype=np.float32)
```

## 5. `get_obs()` の呼び出し箇所を更新

```python
def get_obs(self, agent):
    ...(既存の処理はそのまま)...
    spatial_flat = semantic_obs.reshape(-1)
    action_history_flat = self.last_actions.reshape(-1)

    # 🌟 変更: 最寄りターゲットではなく、割り当てられたターゲットへの方向を使う
    rel_vec = self._compute_relative_direction_to_assigned_prey(agent)

    full_obs = np.concatenate([spatial_flat, action_history_flat, rel_vec])
    return full_obs
```

`self.obs_dim`（`spatial_dim + action_history_dim + direction_dim`）は前回修正済みなので、次元数自体に変更は不要です。

---

## 補足1: 割り当て結果を報酬シェイピングにも活用する（推奨）

観測に方向を与えるだけでなく、既存の索敵報酬（`search_approach_bonus`）も「最寄り」ではなく「割り当てられたターゲット」に近づいたかどうかで評価するよう揃えると、観測と報酬の整合性が取れて学習が安定しやすくなります。

```python
def step(self, agent, action):
    ...(既存)...

    if current_min_dist is None:
        ...
        else:
            # 🌟 変更: 最寄りではなく、割り当てられたターゲットへの距離で評価
            current_global_dist = self._compute_global_dist_to_assigned_prey(agent)
            prev_global_dist = self.prev_global_min_dist.get(agent)
            ...(以下、既存ロジックと同様)...
```

```python
def _compute_global_dist_to_assigned_prey(self, agent) -> float | None:
    raw_env = self.env.unwrapped
    try:
        agent_obj = next(a for a in raw_env.agents if a.name == agent)
        ay, ax = agent_obj.state[1], agent_obj.state[0]
    except Exception:
        return None

    target = self._get_agent_assignment(agent)
    if target is None:
        return None

    ty, tx = target
    return abs(ay - ty) + abs(ax - tx)
```

## 補足2: `order_mode="priority"` の優先度スコアも揃える

以前実装した `compute_priority_scores`（MATのデコード順序決定用）も、同じ割り当て結果を使うよう統一しておくと、「観測」「報酬」「デコード順序」の3つが同じ"担当ターゲット"の概念で一貫します。

```python
def compute_priority_scores(self) -> np.ndarray:
    scores = np.full(self.num_agents, 1e6, dtype=np.float32)
    for i, agent in enumerate(self.possible_agents):
        if agent not in self.env.agents:
            continue
        d = self._compute_global_dist_to_assigned_prey(agent)
        if d is not None:
            scores[i] = d
    return scores
```

---

## 注意点

### 1. 割り当ての不連続な切り替わり（チャタリング）に注意

ハンガリアン法は「今この瞬間の距離コスト」だけで最適化するため、pursuerとpreyが移動するたびに**割り当て先が頻繁に入れ替わる**可能性があります（例えば2体のpreyまでの距離がほぼ同じ場合、わずかな移動で担当が入れ替わり続ける）。これが起きると、エージェントは方向を頻繁に変えさせられ、かえって非効率な動きになりかねません。

気になる場合は、以下のようなヒステリシス（一定期間は前回の割り当てを維持する、または「持ち替えのコスト」を加える）を検討してください。

```python
# 簡易対策案: 一定サイクル数ごとにしか再割り当てしない
def _get_agent_assignment(self, agent):
    current_cycle = getattr(self.env.unwrapped, 'cycles', 0)
    reassignment_interval = 5  # 🌟 5サイクルに1回だけ再計算
    if (self._assignment_cache_cycle == -1 or
            current_cycle - self._assignment_cache_cycle >= reassignment_interval):
        self._assignment_cache = self._compute_agent_target_assignment()
        self._assignment_cache_cycle = current_cycle
    return self._assignment_cache.get(agent)
```

まずは毎サイクル再計算する版で学習し、報酬やcapturesの推移を見て、もし方向の頻繁な切り替わりが問題になっているようならこちらの間引き版を試してください。

### 2. `scipy` の依存追加

`from scipy.optimize import linear_sum_assignment` が実行環境に入っているか確認してください（Google Colab等では標準で利用可能なことが多いですが、念のため）。

### 3. `n_prey < n_agents` のタイル処理について

pursuerの数がpreyの数より多い場合（捕獲が進んで残りprey数が減った終盤など）、コスト行列を横に繰り返して全pursuerに割り当てが行き渡るようにしていますが、これは「余ったpursuerは最も近い(コストの低い)preyに重複して割り当てられやすい」という単純な近似です。厳密に「何人がどのpreyを担当すべきか」を最適化したい場合は、より高度な割り当てロジック（例えば「3人以上必要なpreyには優先的に3人を割り当てる」といった制約付き最適化）が必要になりますが、まずはこのシンプルな版で効果を確認し、必要であれば拡張することをお勧めします。

---

まずはこの実装で数百update学習し、「一角への居座り」「捕獲後の切り替え」の頻度がどう変化するか確認してみてください。割り当てが頻繁に切り替わりすぎているようであれば、上記のヒステリシス対策を追加で検討してください。

ということで変更コードはこちらに保管しています。

https://github.com/Shinichi0713/Reinforce-Learning-Study/tree/main/miulti-agent/petting_zoo/src/4_pursuit/src/mat


## 学習の結果

### 学習の経過

今回のエントロピ・報酬合計・捕獲数のみを抽出して比較した結果をまとめます。

__全体比較（前回 vs 今回）__

| 指標 | 前回 (update 1001-1050) | 今回 (update 1051-1100) | 変化 |
|---|---|---|---|
| **エントロピ（平均）** | 1.4031 | 1.3752 | ▼ 0.0279（低下） |
| **報酬合計（平均）** | 116.22 | 132.37 | ▲ 16.15（改善） |
| **捕獲数（平均）** | 32.0 | 30.7 | ▼ 1.3（微減） |

__読み取れること__

__1. エントロピは継続的に低下__

前回前半 1.4162 → 前回後半 1.3894 → 今回前半 1.3763 → 今回後半 1.3741 と、**方針（policy）が徐々に尖ってきています**。これは探索が減り、特定の行動パターンに収束しつつあることを示しています。

__2. 報酬合計は今回後半で大幅改善__

今回後半の平均報酬 **147.9** は前回平均（116.22）を大きく上回っています。個別エピソードの最大報酬も **223.91**（update 1094）と、前回の最大（176.61）を大幅に更新しています。

__3. 捕獲数はほぼ横ばい__

平均捕獲数は前回 32.0 → 今回 30.7 と微減です。ただし今回後半は 31.8 と回復傾向にあり、最大捕獲数も前回 41 → 今回 42 と同等以上です。

__4. 注意点：今回前半の性能低下__

今回前半（update 1051-1075）は平均捕獲数 **29.7** と低下し、最低 **17**（update 1073）が記録されています。これはチェックポイント読み込み時に `feature_fuse.weight` の形状不一致（64×128 → 64×130）が発生し、**オプティマイザが初期状態にリセットされた影響**と考えられます。

![1787984838933](image/19_global_action_v1/1787984838933.png)

### agentの動作

学習後のエージェントの動作です。
途中まで偵察するエージェントがいる状態で、あるときにエージェントが集まって捕獲に集中している挙動が確認出来ます。
中盤から後半に敵が視界に少なくなってからは場所を移ろうとする気配も確認出来ました。
捕獲数は40体。

期待を満足したわけではありませんが、以前よりも行動はあって欲しい状態へ改善したと考えます。

<img src="image/19_global_action_v1/pursuit_mat_fixed (4).gif">

## 総括

以下、本質を簡潔に総括します。

### 前回問題の本質

MoEの偏りを解消しても、エージェントが一角に居座る問題が残ったのは、**「7×7の局所観測だけでは、視界の外にある敵の方向が分からない」** ためでした。索敵報酬で「動け」と促しても、モデルにとっては「どちらに動いても同じ」という情報不足の状態であり、結果としてリスクの低い狭い範囲の徘徊に収束していました。

### 打った手とその意図

| 対策 | 内容 | 意図 |
|---|---|---|
| **方向情報の追加** | 最寄り未捕獲preyへの相対方向 `(dy, dx)` を観測末尾に連結 | 視界の外の敵の方向をモデルに直接教え、探索の目的を与える |
| **内発的報酬** | 未訪問マスへの count-based exploration bonus を追加 | 敵の手がかりがない状況でも、居座りではなく新しい場所へ動く動機を作る |

### 結果の評価

- **報酬は大幅改善**：今回後半の平均報酬は約148（前回約116）、最大は223.91と記録的な上昇
- **動作に改善の兆候**：「偵察→集団で捕獲→敵が減ったら別エリアへ移動」という流れが確認できた
- **捕獲数は横ばい**：平均30.7と微減。報酬は上がっているが、捕獲効率への転換はまだ不完全
- **エントロピの低下**：方針が尖ってきており、探索と活用のバランスが今後の課題

### 結論

**「敵の方向」という情報を観測に直接与えることで、無目的な徘徊から「意図を持った移動」への転換は実現できました。** ただし、全員が同じ方向情報を見て同じターゲットに集まる「過密化」の問題は残っており、次のステップとして「エージェントごとの役割分担（対策3のハンガリアン法による担当割当）」を検討する価値があるかもしれません。


