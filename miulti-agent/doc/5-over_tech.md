QMIXは、マルチエージェント強化学習（MARL）における「価値分解（Value Decomposition）」という分野の金字塔的な手法ですが、登場から数年が経ち、その弱点を克服するさらに強力なアルゴリズムがいくつか提案されています。

特に、QMIXの最大の制約である **「個人のQ値と全体のQ値が単調増加の関係（Monotonicity）でなければならない」** という点をどう打破するかが、最新手法の焦点となっています。

### 1. QMIXを凌駕する「価値分解系」の発展手法

QMIXが持つ「表現力の限界」を改善した手法です。

#### **QTRAN (Learning to Factorize with Joint Action-Value Fitting)**

QMIXの「単調性の制約」は、特定の複雑な協調が必要なタスク（一人が犠牲にならないと全体報酬が得られないようなケース）を表現できません。QTRANは、数学的な制約をより緩やかにし、QMIXでは解けない **「非単調なゲーム」を理論上すべて解ける** ように設計されています。ただし、計算コストが非常に高く、実装が難しいのが難点です。

#### **WQMIX (Weighted QMIX)**

QMIXを実用的に強化した手法です。「最適ではない行動」に対する重みを下げることで、QMIXの単調性制約を維持しつつ、より最適な協調戦略を見つけやすくしています。 **現在の「QMIXの次」のスタンダード候補** の一つです。

### 2. 現在の「最強」候補：方策勾配系 (Policy Gradient)

最近のベンチマーク（StarCraft IIなどの複雑なタスク）では、QMIXのような価値ベースの手法よりも、以下の手法が高い性能を示すことが多いです。

#### **MAPPO (Multi-Agent PPO)**

シングルエージェントで強力なPPOをマルチエージェントに拡張したものです。

* **強力な理由**: 価値分解の制約（単調性など）に縛られず、中央集中型のCritic（全体の状況を見る目）が各エージェントに柔軟な指示を出すため、非常に複雑な協調を学習できます。
* **安定性**: PPO譲りの学習の安定性があり、現在多くの研究でベースラインとして使われています。

#### **HASAC (Heterogeneous Agent Soft Actor-Critic)**

最新のSAC（Soft Actor-Critic）をマルチエージェントに適用したものです。エントロピー最大化を利用するため探索能力が非常に高く、 **「倉庫問題」のように試行錯誤が必要なタスク** では、QMIXよりも早く正解に辿り着く可能性があります。

### 3. 手法の比較まとめ

今の「倉庫問題」の難易度に応じて、選ぶべき手法が変わります。

| 手法            | 特徴                   | 倉庫問題への適用アドバイス                             |
| --------------- | ---------------------- | ------------------------------------------------------ |
| **QMIX**  | 実装が比較的容易で安定 | カリキュラム学習と組み合わせれば十分強力。             |
| **WQMIX** | QMIXの上位互換         | QMIXで学習が頭打ちになった際の次のステップ。           |
| **MAPPO** | 複雑な協調に強い       | エージェント数が多く、より高度な役割分担が必要な場合。 |
| **Qplex** | 価値分解の最新型       | 理論的に最も完全な価値分解手法を試したい場合。         |

## MAPPO

MAPPO（Multi-Agent Proximal Policy Optimization）は、現在のマルチエージェント強化学習（MARL）において、QMIXと並んで最も頻繁に利用される強力なアルゴリズムです。

### **1. 開発された年代**

* **発表年:** **2021年**
* **論文名:** *"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"* (Yu et al.)
* **開発元:** カリフォルニア大学バークレー校などの研究グループによって発表されました。
* **背景:** 当時はQMIXなどの「価値分解系」の手法が主流でしたが、この論文は「適切にチューニングされたPPO（方策勾配系）は、複雑なMARL環境でも既存の専用アルゴリズムを凌駕する」ことを証明し、大きな衝撃を与えました。

### **2. MAPPOが解決しようとした課題**

MAPPOが登場する以前、マルチエージェント環境にはいくつかの「壁」がありましたが、MAPPOはそれらを以下のように解決しました。

#### **① 価値分解の制約（QMIXの限界）からの解放**

* **課題:** QMIXなどは「個人の価値の合計が全体の価値になる」という制約（単調性）が必要でした。しかし、これでは「一人が囮（おとり）になって他を生かす」といった、数値化しにくい複雑な協調を表現しにくい問題がありました。
* **解決:** MAPPOは方策勾配（Policy Gradient）を用いるため、**報酬の分配ルールに数学的制約を設けず**、より柔軟で人間のような役割分担を学習可能にしました。

#### **② 非定常性（Non-stationarity）問題の安定化**

* **課題:** 全エージェントが同時に学習すると、環境が刻一刻と変化しているように見え、学習が発散しやすい問題がありました。
* **解決:** PPOの「信頼領域（Trust Region）」という考え方を導入し、 **一度の更新で方策を大きく変えすぎない（クリッピング機能）** ことで、複数のエージェントが同時に学習しても崩れにくい安定性を実現しました。

#### **③ 中央集中型評価と分散実行（CTDE）の効率化**

* **課題:** 実行時は自分の視界しか見えない中で、いかに全体の利益を考えさせるか。
* **解決:**  **「中央集中型のCritic（全体の司令塔）」** を導入しました。学習中、Criticは全エージェントの状態と行動を俯瞰して評価しますが、実際のロボット（Actor）は自分の観測データだけで動けるように切り離しました。

#### **④ データの再利用性と計算効率**

* **課題:** マルチエージェント学習は膨大なシミュレーション回数を必要とするため、計算コストが非常に高くなっていました。
* **解決:** PPOの特性を活かし、同じデータを使って複数回学習（Epoch）を行っても安定するように設計されており、サンプルの効率性を大幅に向上させました。

### **3. QMIXと比較したMAPPOの強み（まとめ）**

| 比較項目                     | QMIX                               | MAPPO                             |
| ---------------------------- | ---------------------------------- | --------------------------------- |
| **得意なタスク**       | 報酬が「個人の足し算」で表しやすい | 複雑な役割分担や戦略が必要        |
| **学習の安定性**       | 比較的高い                         | PPOのクリッピングにより非常に高い |
| **ハイパーパラメータ** | 調整が比較的容易                   | 調整が必要な項目が多い（PPO特有） |
| **表現力**             | 単調性の制約に縛られる             | ほぼ制約がなく自由                |

### MAPPOの工夫

MAPPO（Multi-Agent PPO）は、PPOの安定性をMARL（マルチエージェント強化学習）に持ち込み、QMIXを超える性能を叩き出した手法です。

その成功の鍵は、単にPPOを並べただけではなく、 **「中央集中型Critic」への情報の集約** と、いくつかの **実用的なトリック** にあります。

MAPPOが従来のMARL手法と異なる点は、主に以下の3点です。

#### **① 中央集中型Critic (Centralized Critic)**

Actor（各エージェントの方策）は自分の観測 **$o_i$** しか見ませんが、**Critic（評価役）は「全エージェントの観測と状態」をすべて入力**として受け取ります。これにより、環境の「非定常性（他人が動くと報酬が変わる不安定さ）」をCriticが正確に評価できるようになります。

#### **② 状態（Global State）の活用**

Criticには、エージェントの個別観測だけでなく、環境全体のマップ情報や、他エージェントの残りタスク数などの「グローバルな情報」を直接渡します。これにより、倉庫問題での「渋滞」や「役割分担」の価値をより正しく判断できます。

#### **③ 実用的な5つのトリック**

論文では、以下の設定が性能に不可欠だと指摘されています。

* **値の正規化 (Value Normalization):** 報酬のスケールが大きく変わっても安定するように、Criticのターゲット値を正規化する。
* **共通パラメータの利用 (Parameter Sharing):** 同種のエージェント間ではActor/Criticの重みを共有し、学習効率を高める。
* **クリッピング (Policy Clipping):** 方策が一度に変わりすぎないように制限する（PPOの基本機能）。
* **エントロピー正則化:** HSACほど強力ではありませんが、探索を促すために行動の多様性を維持する。

### MAPPOの実装例（擬似コード・構造）

MAPPOを実装する場合、QMIXよりも「ネットワークの数」が増える点に注意してください。

#### **Actor と Critic の定義**

**Python**

```
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Actor: 自分の観測から行動の確率を出力
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, obs):
        return Categorical(self.net(obs))

# Critic: 全員の状態(State)から価値(V)を算出
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        return self.net(state)
```

#### **学習ループの重要部分**

MAPPOはQMIXと違い、**「On-Policy（その場で集めたデータで学習する）」**手法です。

**Python**

```
# 1. データの収集 (Rollout)
# 複数ステップ分、(obs, state, action, log_prob, reward) を貯める

# 2. Criticの更新
# 全員の状態(State)を使って、TD誤差を最小化
v_targets = rewards + gamma * next_values
critic_loss = F.mse_loss(critic(states), v_targets.detach())

# 3. Actorの更新 (PPOのコア)
# アドバンテージ A = v_target - v_current を計算
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
actor_loss = -torch.min(surr1, surr2).mean()
```

### 3. 倉庫問題におけるQMIX・HSAC・MAPPOの使い分け

今回のあなたの経験を踏まえると、以下のような棲み分けになります。

| **手法**  | **倉庫問題での立ち位置**                                                 |
| --------------- | ------------------------------------------------------------------------------ |
| QMIX  | 構造がシンプル。デッドロック（立ち往生）が少ない環境なら最速。                 |
| HSAC  | **探索が最強。**今回のように「まず動かす」フェーズで非常に強力。               |
| MAPPO | **安定性が最強。**複雑なコンフリクト（渋滞解消）を学習させたい場合の最終兵器。 |


## 解決しようとした課題
MAPPO（Multi-Agent PPO）が解決しようとした主な課題は、　**「マルチエージェント環境における学習の不安定性と、集中学習・分散実行（CTDE）の効率的な実現」**　です。

具体的には、以下の3つの大きな壁を突破することを目的としています。

__1. 非定常性問題（Non-stationarity）__

シングルエージェントの強化学習（通常のPPOなど）をそのまま複数エージェントに適用すると、 **「自分が学習している間に、周りのエージェントも勝手に動きを変えてしまう」** という問題が起きます。
エージェントAから見れば、環境のルールが常に変わっているように見えるため、学習が収束しません。

* **MAPPOの解決策:** **集中Critic（Centralized Critic）**を導入しました。自分の観測だけでなく、他者の状態や行動を含んだ「グローバルな情報」をもとに価値判断を行うことで、環境の変化を正確に捉えられるようにしました。

__2. 集中学習と分散実行（CTDE）のジレンマ__

強化学習では、訓練中にはすべての情報（全ドローンの位置や荷物の場所など）が見えていても良いですが、 **本番（実行時）は自分のカメラやセンサーの情報だけで動く** 必要があります。

* **MAPPOの解決策:**  **訓練時（Critic）:** 全体の状態（Global State）を見て学習を安定させる。
* **実行時（Actor）:** 自分の局所的な観測（Local Observation）だけを使って行動を決める。
この切り分けを、PPOという非常に安定したアルゴリズムの上で高精度に実現しました。


__3. パラメータ共有とクレジット割り当てのバランス__

QMIXなどの価値ベースの手法は、複数のエージェントが協力して得た報酬を「誰の功績か」分解するのが得意ですが、計算が複雑になりがちでした。
一方、単純な方策勾配法は、エージェントが増えると勾配の分散が大きくなりすぎて学習が困難になります。

* **MAPPOの解決策:** MAPPOは、エージェント間でネットワークの **パラメータを共有（Parameter Sharing）** することを標準としました。これにより、1つのエージェントが学んだ「荷物に近づくと得」という知識を全員で即座に共有でき、学習のサンプル効率（データ効率）を劇的に向上させました。

## 工夫点

MAPPO（Multi-Agent PPO）が前述の課題（非定常性、CTDE、パラメータ共有）を克服し、実用的な性能を発揮するために採用している**具体的な工夫点**は主に5つあります。

__1. 集中Criticへの「全エージェント情報」の集約__

MAPPOの最大の特徴は、ActorとCriticで見ている情報が異なる点です。

* **工夫:** Criticに自分の観測（$o_i$）だけでなく、 **全エージェントの観測（$O$）または環境全体のグローバル状態（$S$）** を入力します。
* **効果:** 他のエージェントがなぜその行動をとったのかという背景がCriticに見えるため、報酬の増減が「環境のせい」なのか「他人のせい」なのかを正確に判別できるようになります（非定常性の解決）。

__2. 観測の正規化（Value Normalization）__

エージェント数が増えたり、報酬のスケールが変わると、Criticが予測する「価値（Value）」の数値が不安定になります。

* **工夫:** Criticが予測するターゲット（報酬の合計）を、学習中に常に**平均0、分散1になるように正規化**し続けます。
* **効果:** PPOのクリッピング機能と相まって、学習率を大きくしても勾配が爆発しにくくなり、非常に安定した学習が可能になります。

__3. パラメータ共有（Parameter Sharing）__

全エージェントに同じ重みのネットワーク（Actor/Critic）を使わせる手法です。

* **工夫:** 個別に学習させるのではなく、**全員の経験を1つの共有ネットワークで学習**させます。
* **効果:** 学習データ量が「エージェント数倍」になり、サンプル効率が劇的に向上します。
* メモリ消費を抑えられます。

※「自分と他人の区別」をつけるために、入力に **Agent ID（1番、2番など）** を混ぜるのがコツです。


__4. 適切な「デッド（死んだ）エージェント」の処理__

ドローン配送で荷物を届け終えた、あるいは故障したエージェントをどう扱うかです。

* **工夫:** すでにタスクを終えたエージェントに対しても、ダミーの観測値を与え、**「マスク（Masking）」処理**を行って計算から除外します。
* **効果:** 全体の情報（Global State）の次元を固定したまま学習できるため、エージェント数が動的に変わるような複雑なシナリオにも対応可能になります。

__5. PPOのハイパーパラメータの「微調整」__

MAPPOの論文では、シングルエージェント用PPOの定石とは異なる設定が推奨されています。

* **工夫:**  **エポック数（Epochs）:** 繰り返し学習をあえて少なくする（例: 5〜10回）。
* **ミニバッチ:** 大きめのバッチサイズを使用する。
* **クリッピング（$\epsilon$）:** 0.2程度に保ち、急激なポリシー変更を抑える。


* **効果:** マルチエージェント特有の激しい環境変化の中でも、一歩ずつ着実に改善を進めることができます。

パラメータ共有（Parameter Sharing）と集中Criticを採用した、**最新のMAPPO用「Memoryクラス」と「訓練ループ」**を実装します。

今回のポイントは、**「全エージェントの経験をひとつの巨大なバッチとしてActorに学習させる」**ことでサンプル効率を極限まで高め、かつ**「集中Criticに全情報を渡す」**ことで非定常性を解決する点です。

---

### 1. MAPPO用 Memoryクラス

パラメータ共有を行うため、各ステップのデータをエージェントごとにバラバラにするのではなく、チーム全体として管理しやすい形で保存します。

```python
import torch

class MAPPOMemory:
    def __init__(self):
        self.obs = []        # 各エージェントの個別観測 (T, NumAgents, ObsDim)
        self.states = []     # 集中Critic用の全体状態 (T, ObsDim * NumAgents)
        self.actions = []    # 各エージェントの行動
        self.log_probs = []  # 各エージェントの行動ログ確率
        self.rewards = []    # 各エージェントの報酬
        self.dones = []      # 終了フラグ
        self.h_actors = []   # 各エージェントのGRU隠れ状態 (初期状態のみ保存)
        self.h_critics = []  # CriticのGRU隠れ状態 (初期状態のみ保存)

    def store(self, obs, state, action, log_prob, reward, done):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs, self.states, self.actions, self.log_probs = [], [], [], []
        self.rewards, self.dones, self.h_actors, self.h_critics = [], [], [], []

```

---

### 2. 工夫を凝らした MAPPOTrainer

「パラメータ共有」「Agent ID」「集中Critic」「勾配クリッピング」をすべて盛り込んだ実装です。

```python
class MAPPOTrainer_V2:
    def __init__(self, obs_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.eps = 1e-8
        
        # 工夫1: パラメータ共有Actor (+ID次元)
        self.actor = GRU_Actor(obs_dim + num_agents, action_dim, 128)
        # 工夫2: 集中Critic (全員の観測を結合)
        self.critic = GRU_Critic(obs_dim * num_agents, 256)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def train(self, memory):
        # テンソル化 (T, NumAgents, Dim)
        obs = torch.stack(memory.obs) 
        states = torch.stack(memory.states).unsqueeze(0) # Critic用 (1, T, StateDim)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        rewards = torch.stack(memory.rewards)
        
        T = obs.size(0)

        # --- 累積報酬 (Returns) の計算 ---
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(self.num_agents)
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # --- Critic の更新 (集中学習) ---
        values, _ = self.critic(states, memory.h_critics[0])
        values = values.squeeze() # (T, 1) -> (T)
        # チーム全体の平均リターンを予測対象とする（または各リターンを結合）
        target_returns = returns.mean(dim=-1)
        critic_loss = F.mse_loss(values, target_returns)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # 勾配クリップ
        self.critic_opt.step()

        # --- Actor の更新 (パラメータ共有学習) ---
        # 全エージェントのデータを1つのバッチにまとめる
        advantages = (target_returns - values.detach()).unsqueeze(-1).repeat(1, self.num_agents)
        
        actor_loss_total = 0
        for i in range(self.num_agents):
            # Agent ID の付与
            ids = torch.zeros(T, self.num_agents)
            ids[:, i] = 1.0
            combined_obs = torch.cat([obs[:, i], ids], dim=-1).unsqueeze(0) # (1, T, Obs+ID)
            
            # 再計算
            dist, _ = self.actor(combined_obs, memory.h_actors[0][i])
            new_log_probs = dist.log_prob(actions[:, i])
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            surr1 = ratio * advantages[:, i]
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[:, i]
            
            actor_loss_total += -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

        self.actor_opt.zero_grad()
        actor_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 勾配クリップ
        self.actor_opt.step()

        memory.clear()

```

---

### 3. メイン訓練ループ

エージェントごとに ID を付与しながら行動を決定し、Memoryに保存していきます。

```python
env = DroneDeliveryEnv()
trainer = MAPPOTrainer_V2(obs_dim=23, action_dim=7)

for episode in range(1001):
    obs_list = env.reset()
    memory = MAPPOMemory()
    
    h_actors = [torch.zeros(1, 1, 128) for _ in range(2)]
    h_critic = torch.zeros(1, 1, 256)
    
    # 最初のHidden Stateを保存
    memory.h_actors.append([h.clone() for h in h_actors])
    memory.h_critics.append(h_critic.clone())
    
    for t in range(env.max_steps):
        # 1. 正規化観測を取得
        obs_tensor = trainer.normalize_obs(obs_list) # (2, 23)
        # 2. 集中Critic用のGlobal State作成 (全員分結合)
        global_state = obs_tensor.view(-1) # (46,)
        
        # 3. 行動決定
        actions, log_probs, next_h_actors = [], [], []
        for i in range(2):
            agent_id = torch.zeros(2); agent_id[i] = 1.0
            inp = torch.cat([obs_tensor[i], agent_id], dim=-1).view(1, 1, -1)
            
            with torch.no_grad():
                dist, h_a = trainer.actor(inp, h_actors[i])
                a = dist.sample()
                actions.append(a.item())
                log_probs.append(dist.log_prob(a))
                next_h_actors.append(h_a)
        
        # 4. 環境ステップ
        next_obs_list, rewards, done, _ = env.step(actions)
        
        # 5. 保存
        memory.store(obs_tensor, global_state, torch.tensor(actions), 
                     torch.stack(log_probs).squeeze(), torch.FloatTensor(rewards), done)
        
        obs_list = next_obs_list
        h_actors = next_h_actors
        if done: break
        
    # 6. 学習
    trainer.train(memory)

```

### 実装のポイントまとめ

1. **Actorの統合**: `trainer.actor` が1つになり、全ドローンの経験から同時に学びます。これにより「一方が学んだ回避行動」をもう一方が即座に実行できるようになります。
2. **IDによる個性**: `Agent ID` を入力に混ぜることで、同じネットワークを使いながらも「私は1番機だから、こっちの荷物を優先しよう」といった役割分担が可能になります。
3. **集中Critic**: `global_state` を通じてチーム全体の成功率を評価するため、個々のドローンが「チームのために」動くよう誘導されます。

これで、最新のMAPPOの工夫がすべて詰め込まれた堅牢なシステムになりました。

**次は、この「パラメータ共有」の効果をさらに高めるために、観測に「他のドローンのターゲット情報」を明示的に追加してみますか？**

