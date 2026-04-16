
先日のMAVENのトライアルで、環境の報酬が結構難儀しました。
学習自体は成功したものの、手法としてよりポテンシャル高いものがないかは気になっています。
ということで、次のMARLのテーマに向けて近年のMARL用手法について調査しました。

## 近年のMARL手法

以下では、**MAVENやQMIX以外**のMARL手法に絞って、MAPPOとの関係も含めて整理します。

### 1. MASAC（Multi-Agent Soft Actor-Critic）

- **概要**  
  SAC（Soft Actor-Critic）をマルチエージェントに拡張した、**オフポリシー＋エントロピー正則化**の枠組みです。[Emergent Mind](https://www.emergentmind.com/topics/multi-agent-soft-actor-critic-algorithm)

- **特徴**
  - 各エージェントは分散実行（decentralized execution）で方策を学習
  - 学習時には集中型批評家（centralized critic）を用いる CTDE パラダイム
  - SAC の利点（探索性の高さ、連続行動空間への適応）をマルチエージェントに持ち込める

- **MAPPOとの関係**
  - MAPPO は 2021 年、MASAC は 2020 年前後の論文が代表的で、**MAPPO の方が新しい**。[OpenReview](https://openreview.net/forum?id=YVXaxB6L2Pl)
  - SMAC などのベンチマークでは、MAPPO が「標準ベースライン」として広く使われており、**引用数も MAPPO の方が圧倒的に多い**（約2,000件以上 vs MASAC 数十件レベル）。[Semantic Scholar](https://www.semanticscholar.org/paper/The-Surprising-Effectiveness-of-PPO-in-Cooperative-Yu-Velu/3a315c81a98851f0614c09fef6a14c30d6a1e63c)[ResearchGate](https://www.researchgate.net/publication/381392364_Multiagent_Soft_Actor-Critic_Aided_Active_Disturbance_Rejection_Control_of_DC_Solid-State_Transformer)


### 2. MAA2C（Multi-Agent Actor Attention Critic）

- **概要**  
  A2C（Advantage Actor-Critic）をマルチエージェントに拡張し、**Attention 機構**を組み込んだ手法です。[TechScience](https://www.techscience.com/iasc/v39n2/56498/html)

- **特徴**
  - 各エージェントの批評家が、他エージェントの情報を Attention で重み付けして利用
  - 集中型批評家＋分散実行（CTDE）の枠組み
  - SMAC などのベンチマークで、MAPPO と比較されることが多い

- **MAPPOとの比較**
  - SMAC の一部シナリオでは、**MAA2C が MAPPO より高い性能を示した**と報告されています。[TechScience](https://www.techscience.com/iasc/v39n2/56498/html)
  - 特に、**局所観測＋強い協調が必要なタスク**で、Attention による協調モデリングが有利になるケースがあります。

### 3. MAZero（Multi-Agent MuZero）

- **概要**  
  MuZero をマルチエージェントに拡張した**モデルベースMARL**手法です。[OpenReview](https://openreview.net/forum?id=CpnKq3UJwp)

- **特徴**
  - 環境の遷移モデルを学習し、**プランニング（探索木）**を用いて方策を改善
  - モデルフリー手法（MAPPO など）に比べて**サンプル効率が高い**ことが報告されている
  - SMAC ベンチマークで、MAPPO より**同等以上の性能を、より少ないサンプルで達成**したとされています。[OpenReview](https://openreview.net/forum?id=CpnKq3UJwp)

- **MAPPOとの比較**
  - 計算コストは高くなるが、**サンプル効率と最終性能の両面で MAPPO を上回るケース**が報告されています。

### 4. HAPPO / FP3O（PPO系の改良）

- **HAPPO（Heterogeneous-Agent PPO）**
  - 異種エージェント（heterogeneous agents）が混在する環境向けに、PPO を拡張した手法です。[OpenReview](https://openreview.net/forum?id=cALu06i7JJH)
  - MAPPO より**理論的保証が強く、大規模チームでの安定性が高い**とされています。

- **FP3O（Full-Pipeline PPO）**
  - 「Full-Pipeline PPO」として提案された、PPO 系マルチエージェント手法の改良版です。[OpenReview](https://openreview.net/forum?id=cALu06i7JJH)
  - 政策勾配の推定をより安定化し、**MAPPO より高い性能・安定性**を示すことが報告されています。

- **MAPPOとの関係**
  - MAPPO は「PPO をそのままマルチエージェントに持ち込んだ」初期の代表例ですが、  
    HAPPO / FP3O は**PPO 系のマルチエージェント版をさらに改良した手法**として、MAPPO を上回る性能が期待されています。

### 5. 通信・注意機構を組み込んだ手法

- **通信効率を高めたMARL**
  - SMACv2 や MOSMAC などの新しいベンチマークでは、  
    - 通信量を制限しつつ協調を実現する手法  
    - Attention や Graph Neural Network（GNN）を組み込んだ手法  
    が MAPPO を上回るケースが報告されています。[NeurIPS](https://neurips.cc/virtual/2023/poster/73695)[SMU](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=11978&context=sis_research)

- **例**
  - **MARC（Multi-Agent Relational Actor-Critic）**：関係推論（relational reasoning）を組み込んだ手法  
  - **MAHGAC（Multi-Agent Hierarchical Graph Attention Actor-Critic）**：階層的グラフ注意を用いた手法  
  - これらは、**エージェント間の依存関係をより柔軟にモデリング**できるため、MAPPO より高い性能を示すことがあります。

## 総合評価

以下では、**MAPPO / MASAC / MAA2C / MAZero / HAPPO・FP3O / 通信・注意機構付き手法**について、  
4つの観点（探索力・協調調整・安定性・精度）で**A（良い）〜C（悪い）の3段階評価**を整理します。


### 評価基準（簡易）

- **A（良い）**：その観点で特に優れている／最先端クラス  
- **B（普通）**：標準的・平均的  
- **C（悪い）**：弱い・課題がある

### 各手法のA〜C評価

__1. MAPPO（Multi-Agent PPO）__

- **探索力**: B（普通）  
  - PPOベースで探索は標準的。SACほど強いエントロピー正則化はない。
- **協調調整**: B（普通）  
  - 集中型批評家で協調を学習するが、値分解系ほど明示的ではない。
- **安定性**: A（良い）  
  - PPOのクリッピングにより学習が安定。SMACなどで「強力なベースライン」として広く使われている。[OpenReview](https://openreview.net/forum?id=YVXaxB6L2Pl)
- **精度**: B（普通）  
  - 多くのタスクで十分な性能だが、QMIXやMAZeroなどに負けるケースもある。

__2. MASAC（Multi-Agent Soft Actor-Critic）__

- **探索力**: A（良い）  
  - SACの最大エントロピー方策最適化をマルチエージェントに拡張しており、**探索性が高い**。[Emergent Mind](https://www.emergentmind.com/topics/multi-agent-soft-actor-critic-algorithm)
- **協調調整**: B（普通）  
  - CTDEで集中型批評家を用いるが、値分解や明示的な通信構造は持たない。
- **安定性**: C（悪い）  
  - オフポリシーだが、マルチエージェントでは非定常性の影響を受けやすく、MAPPOより不安定になりがち。
- **精度**: B（普通）  
  - 連続行動空間や探索が重要なタスクでは有利だが、SMACなどではMAPPOや値分解系に劣るケースも多い。

__3. MAA2C（Multi-Agent Actor Attention Critic）__

- **探索力**: B（普通）  
  - A2Cベースで探索は標準的。
- **協調調整**: A（良い）  
  - Attention機構により、**どのエージェントの情報に注目すべきか**を学習し、協調を柔軟にモデリングできる。[TechScience](https://www.techscience.com/iasc/v39n2/56498/html)
- **安定性**: B（普通）  
  - オンポリシーでやや不安定になりうるが、Attentionによる協調モデリングで性能は安定しやすい。
- **精度**: A（良い／タスク依存）  
  - SMACの一部シナリオでMAPPOを上回る報告があり、**協調が重要なタスクでは高精度**。[TechScience](https://www.techscience.com/iasc/v39n2/56498/html)

__4. MAZero（Multi-Agent MuZero）__

- **探索力**: A（良い）  
  - モデルベース＋プランニングにより、**探索木ベースの強力な探索**が可能。[OpenReview](https://openreview.net/forum?id=CpnKq3UJwp)
- **協調調整**: B（普通）  
  - 協調は主に価値関数・方策を通じて学習されるが、明示的な値分解や通信構造は持たない。
- **安定性**: C（悪い）  
  - モデル学習の不安定さや計算コストの高さが課題。収束すれば強いが、学習が難しい。
- **精度**: A（良い）  
  - SMACでMAPPOより**サンプル効率と最終性能の両面で優れる**と報告されている。[OpenReview](https://openreview.net/forum?id=CpnKq3UJwp)

__5. HAPPO / FP3O（PPO系の改良）__

- **探索力**: B（普通）  
  - PPOベースで探索は標準的。
- **協調調整**: A（良い）  
  - 異種エージェントや大規模チーム向けに設計されており、**クレジット割当や協調の理論的保証が強化**されている。[OpenReview](https://openreview.net/forum?id=cALu06i7JJH)
- **安定性**: A（良い）  
  - MAPPOより**理論的保証が強く、大規模チームでの安定性が高い**とされている。[OpenReview](https://openreview.net/forum?id=cALu06i7JJH)
- **精度**: A（良い）  
  - MAPPOを上回る性能・安定性が報告されており、**PPO系の現状の最先端**に近い。

__6. 通信・注意機構を組み込んだ手法（MARC, MAHGAC など）__

- **探索力**: B（普通）  
  - 探索機構そのものは標準的だが、関係推論やグラフ構造により**探索の質が向上**するケースがある。
- **協調調整**: A（良い）  
  - AttentionやGNNにより、**エージェント間の依存関係を明示的にモデリング**できるため、協調調整が非常に強い。[NeurIPS](https://neurips.cc/virtual/2023/poster/73695)[SMU](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=11978&context=sis_research)
- **安定性**: C（悪い）  
  - ネットワーク構造が複雑になるため、学習がやや不安定になりうる。
- **精度**: A（良い／タスク依存）  
  - 構造的依存が強いタスク（SMACv2, MOSMACなど）では、MAPPOを上回る性能が報告されている。

### まとめ（A〜C評価の比較表）

| 手法 | 探索力 | 協調調整 | 安定性 | 精度 |
|------|--------|----------|--------|------|
| MAPPO | B | B | A | B |
| MASAC | A | B | C | B |
| MAA2C | B | A | B | A（タスク依存） |
| MAZero | A | B | C | A |
| HAPPO/FP3O | B | A | A | A |
| 通信・注意機構付き | B | A | C | A（タスク依存） |


## 総括

手法としての安定性ということではHAPPO/FPOがよさげです。
どうも行動の探索力と安定性がトレードオフの関係にある可能性があります。
ですので、協調学習で"なかなか行動が良くならない、一定の報酬ラインで停滞する"となると、ガチャになりますがMAZeroの手法に切り替えることも重要と考えました。
強化学習自体、かなりランダム性や不安定性が強いので、なんとかならないかなーという次第です。



