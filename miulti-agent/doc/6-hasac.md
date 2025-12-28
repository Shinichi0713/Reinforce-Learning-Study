HASAC（Heterogeneous Agent Soft Actor-Critic）は、マルチエージェント強化学習（MARL）において、**「各エージェントの能力や役割が異なる環境（不均一なエージェント集合）」**でも、効率的かつ安定的に協調を学習させるために提案された最新の手法の一つです。

この手法は、シングルエージェントで非常に強力な **SAC (Soft Actor-Critic)** をベースに、マルチエージェント特有の課題を解決するために拡張されました。


## HASACが解決しようとした課題

従来のマルチエージェント手法（QMIXやMAPPOなど）には、以下の課題がありました。

* **同時更新による不安定性（Non-stationarity）**:
全エージェントが同時に自分の方策（Policy）を更新すると、環境が刻一刻と変化しているように見え、学習が発散しやすくなります。
* **探索の不足**:
特に倉庫問題のように「複雑な手順を踏まないと報酬が得られない」タスクでは、従来の決定論的な手法（QMIXなど）は局所解（誰も動かなくなるなど）に陥りやすい傾向がありました。
* **不均一なエージェント（Heterogeneity）への対応**:
多くの手法は「全エージェントが同じ能力を持つ」ことを前提としたパラメータ共有を行いますが、現実には「足の速いロボット」と「荷物を持てるロボット」のように役割が異なる場合があり、これらを一括で学習するのは非効率でした。


## HASACの主な特徴

HASACは、これらの課題を解決するために以下の仕組みを導入しています。

#### ① 最大エントロピー強化学習 (Soft Actor-Criticの継承)

SACの最大の特徴である「エントロピー正則化」を利用します。

* **意味**: 報酬を最大化するだけでなく、 **「行動の多様性（エントロピー）」も最大化** するように学習します。
* **メリット**: 「とりあえず色々な動きを試してみる」という探索が強力に行われるため、倉庫のような複雑なグリッド環境でも、デッドロックを回避する新しい経路を見つけやすくなります。

#### ② 逐次的な方策更新 (Sequential Policy Update)

HASACの最もユニークな点は、エージェントが**一人ずつ順番に**方策を更新していく仕組みです。

* **仕組み**: 数学的な「マルチエージェント・アドバンテージ分解補題」に基づき、エージェント1が更新し、その新しい方策を前提にエージェント2が更新する、という流れを保証します。
* **メリット**: 全員が同時に勝手な動きを変えることがなくなるため、理論的に**「方策の単調改善（Monotonic Improvement）」**、つまり学習を繰り返すほど必ずチームの性能が上がることが保証されます。

#### ③ 中央集中型Criticと分散型Actor (CTDE)

学習時は「全エージェントの状態と行動」を俯瞰して評価する強力な **Critic** を使い、実行時は自分の観測だけで動く **Actor** を使います。


## QMIXと比較した時のメリット

一番気になるこれまでダメだったQMIXとの違いを整理してみました。
今回の問題は荷物を取って、目的地に運搬するという複合動作なので表現力不足になっている可能性はありました。
後は学習の安定性にはかなり難があったと思います。

| 特徴 | QMIX | HASAC |
| --- | --- | --- |
| **表現力** | 単調性（Monotonicity）の制約がある | 制約がなく、より複雑な協調を表現可能 |
| **探索能力** | 決定論的（-greedyに依存） | 確率論的（エントロピー最大化による強力な探索） |
| **安定性** | TD誤差による学習で比較的安定 | 逐次更新により理論的な改善が保証される |
| **不均一性** | パラメータ共有が前提になりやすい | 異なる能力のエージェント混在に強い |



## HASACの実装
HASAC（Heterogeneous Agent Soft Actor-Critic）の実装は、シングルエージェントのSACを「中央集中型学習・分散実行（CTDE）」の枠組みに拡張し、さらにエージェント間の「逐次的な更新」を組み込むことで実現されます。

QMIXに比べると、各エージェントが独立した方策（Policy）を持つため、コードの複雑さは増しますが、その分柔軟な協調が可能になります。


### 1. HASACの全体構造

HASACは以下の3つのコンポーネントで構成されます。

1. **Shared Critic (集中Critic)**: 全エージェントの状態  と行動集合  を入力とし、チーム全体の  値を評価します。
2. **Individual Actors (分散Actor)**: 各エージェント  が、自分の観測  に基づいて行動の確率分布（平均と分散）を出力します。
3. **Alpha (温度パラメータ)**: 探索（エントロピー）の重要度を調整します。


### 2. ニューラルネットワークの定義

PyTorchを用いた基本的なネットワークの構成です。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 各エージェントの方策ネットワーク
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.fc(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2) # 数値安定化
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample() # reparameterization trick
        # 離散アクションの場合は Gumbel-Softmax 等を使用
        return x, dist.log_prob(x).sum(dim=-1, keepdim=True)

# チーム全体のQ値を評価するネットワーク
class Critic(nn.Module):
    def __init__(self, state_dim, all_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # 全員の状態と全員の行動を結合して入力
        self.fc = nn.Sequential(
            nn.Linear(state_dim + all_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.fc(x)

```

### 3. 学習（更新）のアルゴリズム

HASACの最大の特徴である「逐次的更新」を実装する際のロジックです。

#### ① Criticの更新

通常のSACと同様に、ターゲットネットワークを用いてTD誤差を最小化します。


#### ② Actorの更新（ここがHASACの独自ポイント）

エージェント  を更新する際、自分より前のインデックスのエージェント  は新しい方策を、後のエージェント  は古い（サンプリング時の）方策を使うように設計します。

```python
def update_actors(self, batch):
    states, obs_list, actions_list, rewards, next_states, dones = batch
    
    # 全エージェントのアクションを最新の方策でサンプリングし直す（逐次更新の準備）
    current_actions = []
    current_log_probs = []
    for i in range(self.n_agents):
        action, log_prob = self.actors[i].sample(obs_list[i])
        current_actions.append(action)
        current_log_probs.append(log_prob)

    # 各エージェントごとに順番にロスを計算
    for i in range(self.n_agents):
        # 他のエージェントの行動を固定し、エージェントiの行動だけを微分対象にする
        joint_actions = torch.cat([
            # 0〜i-1までは新しい行動、iは現在の微分対象、i+1以降はサンプリング時の行動
            # (簡易的には全員新しい行動を入れても学習は回りますが、HASACの厳密解は逐次です)
            *current_actions[:i+1], *actions_list[i+1:]
        ], dim=-1)
        
        q_values = self.critic(states, joint_actions)
        
        # SACの目的関数: Q値 + α * エントロピー
        actor_loss = (self.alpha * current_log_probs[i] - q_values).mean()
        
        self.actor_optimizers[i].zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizers[i].step()

```

### 4. 倉庫問題（離散アクション）への適用アドバイス

提供いただいたコードのアクションは「待機、上、下、左、右」の5つの離散値です。SACやHASACを離散空間で実装する場合は、以下のいずれかの手法をとります。

1. **Gumbel-Softmax**: カテゴリカル分布を微分可能にする手法。
2. **Discrete SAC**: Actorから各アクションの確率  を出力し、Q値との期待値をとる手法（こちらの方が安定します）。

### 5. 実装へのロードマップ

1. **集中状態（State）の構築**: QMIXで使っていた `state_vec` をそのまま `Critic` の入力に使います。
2. **Actorの複製**: エージェントごとに `Actor` インスタンスを作成します（不均一性を許容するため）。
3. **リプレイバッファ**: `(s, o_1, o_2, a_1, a_2, r, s', done)` を保存するように拡張します。

今回、エージェントが学習後、動作しないような現象が結構見られました。
今回実装予定のHASACはQMIXに比べ、**「デッドロック（お互い動かなくなる状態）」をエントロピーによる探索で打破しやすい**のが強みです。




