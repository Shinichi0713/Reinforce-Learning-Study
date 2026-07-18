# 再イメージング状態レイアウトデザインwith DRL

SLDは中心的な役割を果たす。

レーザーウォールが直進の壁か曲がった壁であるべき。
フロアプランに壁の基礎が形作られた時に、セグメントから放たれる光が既存の壁がソフトパートまでか、ハードパート。

## 状態空間
RLはエージェントの行動背景となる。
累積的な報酬を最大化することを狙って、エージェントは行動を行う。
状態空間は、２つの観点で感知される。エージェントの観点と、環境の観点です。

このようにして、エージェントと環境は、意思決定するための状態の知識が必要となる。
スペースレイアウトは、離散セルの２次元により構成される。
初めに、画像の特徴量を通して、NNに通す前にグレースケースに加工されている。
次に、特徴量ベクトルを通して、どの壁も、そのほかの壁も


## 実験

CNNのレイヤを熱くした状態で、繰り返し学習した場合に報酬がどのように推移するか確認する。

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),
    nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
    nn.Flatten()
)
```

```
Episode 0: Total reward = 13.0
Episode 1: Total reward = 20.0
Episode 2: Total reward = 12.0
Episode 3: Total reward = 38.0
Episode 4: Total reward = 16.0
```

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
    nn.Flatten()
)
```

```
Episode 0: Total reward = 23.0
Episode 1: Total reward = 30.0
Episode 2: Total reward = 10.0
Episode 3: Total reward = 11.0
Episode 4: Total reward = 17.0
```

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),          # 3x3: 基本的な特徴抽出
    nn.Conv2d(16, 32, 1, stride=1), nn.ReLU(),         # 1x1: チャネル変換
    nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.ReLU(), # 5x5: 広い特徴抽出（パディングでサイズ調整）
    nn.Conv2d(32, 32, 1, stride=1), nn.ReLU(),         # 1x1: チャネル変換
    nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), # 3x3: 追加の特徴抽出
    nn.Flatten()
)
```


```
Episode 0: Total reward = 19.0
Episode 1: Total reward = 41.0
Episode 2: Total reward = 20.0
Episode 3: Total reward = 40.0
Episode 4: Total reward = 19.0
```

CNNのレイヤのサイズのせいではない。  
入力画像のサイズを変更して、変化が出るかを確認する。

入力画像のサイズを５倍

```
Episode 0: Total reward = 12.0
Episode 1: Total reward = 10.0
Episode 2: Total reward = 14.0
Episode 3: Total reward = 41.0
Episode 4: Total reward = 20.0
```

変化なし。

## 推測精度
後ほど確認すると、CNNによる推測は精度が良くないとのこと。

---
とても良い質問です。  
「部屋の間仕切りをCNNを使ったRLで学習するAIモデル」＝「部屋の構造や間仕切り（壁など）を画像で認識し、強化学習で行動を学ぶAI」を想定します。

---

## 1. 画像入力（CNN+RL）で精度は出るか？

### 結論
- **簡単な部屋や間仕切りなら、十分なデータ・学習時間・適切なCNN設計があれば、画像入力でも精度を出すことは可能です。**
- しかし、**画像入力は「状態空間が広く、特徴抽出が難しい」ため、学習が難航したり、精度が安定しなかったりすることが多いです。**
- 特に複雑な間取りや照明・視点の変化・ノイズがある場合は、かなり大規模なモデルやデータが必要になります。

### 理由
- 画像から壁や出入口などの「意味的な構造」を認識し、かつ最適行動を学ぶ必要があるため、「特徴抽出」と「方策学習」の両方を同時に行うことになり、難易度が高いです。
- 画像認識部分がうまくいかないと、強化学習もうまく進みません。

---

## 2. 画像以外での状態表現のアイデア

### 実際のロボットやシミュレータでは、画像以外の状態取得も有効です。

### 代表的な方法

#### (1) センサー値や座標を直接使う
- 位置（x, y）、向き（θ）、距離センサー値（壁までの距離）などをstateとして直接使う
- これらは数値ベクトルなので、MLP（全結合層）で十分高精度に学習できる
- ロボットの自己位置推定（SLAM）やレーザー距離計（LIDAR）値なども有効

#### (2) 部屋の地図やグリッドマップ
- 部屋全体をグリッド状（0:空き, 1:壁など）に離散化し、状態として与える
- 例：エージェントの現在位置＋周囲8セルの壁有無など
- これもCNNで処理可能（2Dグリッドは画像に似ている）

#### (3) オブジェクト情報・抽象化された特徴
- 壁やドア、障害物の位置情報を抽出し、数値ベクトルとして与える
- 例：最近傍の壁までの距離、ドアまでの方向・距離など

#### (4) 複数センサーの組み合わせ
- 画像＋距離センサー値
- 画像＋自己位置推定値
- など、マルチモーダルな状態表現

---

## 3. まとめ

- 画像のみを使う場合、難易度は高いですが、十分な設計・リソースがあれば精度は出せます（ただしstateベクトル入力より難しい）。
- **画像以外の状態表現（数値ベクトル、グリッドマップ、センサー値など）を使うと、学習が容易になり、精度も高くなりやすいです。**
- 状態設計を工夫することで、より効率的で高精度な強化学習が可能になります。

---

### まとめ例

> **部屋の間仕切り問題をCNN＋RLで画像入力のみで解くことは可能ですが、学習効率や精度の面では難易度が高いです。  
> 画像以外にも、位置やセンサー値、グリッドマップなどの「数値ベースの状態表現」を使うことで、より高精度かつ効率的に学習できる可能性が高いです。**


## 採択論文一覧

「ここ２年」を 2025〜2026 年の主要国際会議（ICLR, NeurIPS, ICML）と解釈し、強化学習（Reinforcement Learning）関連の採択論文タイトルを会議ごとに整理しました。

---

## ICLR 2025（2025年開催）

ICLR 2025 の公式ページから、強化学習関連の代表的なタイトルを一部抜粋します。

- **Training Language Models to Self-Correct Via Reinforcement Learning**  
  （大規模言語モデルを強化学習で自己修正させる手法）  
  [ICLR 2025 Papers](https://iclr.cc/virtual/2025/papers.html)

※ICLR 2025 のRL関連タイトルは、公式ページ全体から「Reinforcement Learning」をキーワードに抽出する必要がありますが、ここでは代表例のみを挙げています。  
詳細な一覧は上記公式ページの検索機能で「reinforcement learning」などで絞り込むと確認できます。

---

## NeurIPS 2025（2025年開催）

NeurIPS 2025 の公式採択論文リストから、強化学習（Reinforcement Learning）関連のタイトルを抜粋します[NeurIPS 2025 Papers](https://neurips.cc/virtual/2025/papers.html)。

- Flow-Based Policy for Online Reinforcement Learning
- Beyond Scalar Rewards: An Axiomatic Framework for Lexicographic MDPs
- Tru-POMDP: Task Planning Under Uncertainty via Tree of Hypotheses and Open-Ended POMDPs
- Cognitive Predictive Processing: A Human-inspired Framework for Adaptive Exploration in Open-World Reinforcement Learning
- Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm
- Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization
- Pessimistic Data Integration for Policy Evaluation
- CHPO: Constrained Hybrid-action Policy Optimization for Reinforcement Learning
- UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection
- Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning
- Computational Hardness of Reinforcement Learning with Partial q^pi-Realizability
- Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning
- Reward-Aware Proto-Representations in Reinforcement Learning
- Explainably Safe Reinforcement Learning
- FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning
- Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations
- Optimal Single-Policy Sample Complexity and Transient Coverage for Average-Reward Offline RL
- REINFORCEMENT LEARNING FOR INDIVIDUAL OPTIMAL POLICY FROM HETEROGENEOUS DATA
- Meta-World+: An Improved, Standardized, RL Benchmark
- Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
- COGNAC: Cooperative Graph-based Networked Agent Challenges for Multi-Agent Reinforcement Learning
- Risk-Averse Total-Reward Reinforcement Learning
- Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design

---

## ICML 2025（2025年開催）

ICML 2025 の「Reinforcement Learning」セクションに掲載されているポスター論文タイトルは以下の通りです[ICML 2025 Papers (Reinforcement Learning section)](https://github.com/DmitryRyumin/ICML-2025-Papers/blob/main/sections/2025/main/reinforcement-learning.md)。

- Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration
- Cross-Environment Cooperation Enables Zero-Shot Multi-Agent Coordination
- Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning
- Temporal Difference Flows

---

## ICLR 2026（2026年開催）

ICLR 2026 の公式採択論文リストから、強化学習関連のタイトルを抜粋します[ICLR 2026 Papers](https://iclr.cc/virtual/2026/papers.html)。

- Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning
- Scalable Offline Model-Based RL with Action Chunks
- Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead
- TROLL: Trust Regions Improve Reinforcement Learning for Large Language Models
- WebArbiter: A Generative Reasoning Process Reward Model for Web Agents
- Multi-Bellman operator for convergence of Q-learning with linear function approximation
- Information Theoretic Guarantees For Policy Alignment In Large Language Models
- JustRL: Scaling a 1.5B LLM with a Simple RL Recipe
- Learning to summarize user information for personalized reinforcement learning from human feedback
- From REINFORCE to Dr. GRPO: A Unified Perspective on LLM Post-Training
- BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning
- Is the evidence in 'Language Models Learn to Mislead Humans via RLHF' valid?
- DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty
- OPPO: Accelerating PPO-based RLHF via Pipeline Overlap
- LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards
- EXPO: Stable Reinforcement Learning with Expressive Policies
- What Matters for Batch Online Reinforcement Learning in Robotics?
- DEAS: DEtached value learning with Action Sequence for Scalable Offline RL
- MIRACLE: Model-free Imitation and Reinforcement Learning for Adaptive Cut-Selection
- SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
- Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs
- Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning
- Effect of Parallel Environments and Rollout Steps in PPO
- Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?
- Who Matters Matters: Agent-Specific Conservative Offline MARL
- SafeMPO: Constrained Reinforcement Learning with Probabilistic Incremental Improvement
- Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild
- Learning to Maximize Rewards via Reaching Goals
- KL-Regularized Reinforcement Learning for Generative Modelling is Designed to Mode Collapse
- MolEditRL: Structure-Preserving Molecular Editing via Discrete Diffusion and Reinforcement Learning
- Learning Nonlinear Causal Reductions to Explain Reinforcement Learning Policies
- Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models
- Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning
- Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals
- Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning
- Critique-RL: Training Language Models For Critiquing Through Two-Stage Reinforcement Learning
- AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL
- BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping
- Scaling Goal-conditioned Reinforcement Learning with Multistep Quasimetric Distances
- ExGRPO: Learning to Reason from Experience
- Using Graph Neural Networks in Reinforcement Learning: A Practical Guide
- Causally Robust Reward Learning from Reason-Augmented Preference Feedback
- Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning
- Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function

---

## 補足

- 上記は「強化学習」を明示的に扱う論文タイトルを中心に抜粋したものです。  
- 実際には、各会議の公式ページ（ICLR, NeurIPS, ICML）で「reinforcement learning」や「RL」で検索すると、さらに多くの関連論文タイトルを確認できます。
- もし特定のサブトピック（例：オフラインRL、マルチエージェントRL、RLHF など）に絞ったタイトル一覧が必要でしたら、その旨お知らせいただければ、該当分野に特化したリストも作成いたします。

