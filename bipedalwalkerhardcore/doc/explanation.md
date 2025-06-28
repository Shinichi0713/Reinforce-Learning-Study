## 目的



## ランダム操作
とりあえず動作確認。
![alt text](image.png)

## この問題の特徴
BipedalWalkerHardcoreは、連続値の行動空間を持つ高難易度のロボティクス環境であり、ノイズや不安定さ、複雑な報酬設計が特徴です。
- 連続値の行動空間
- ノイズや不安定さ
- 複雑な報酬設計

## 強化学習フレームワークの検討

そのため、以下の強化学習アルゴリズムが有力です。


### 1. **Proximal Policy Optimization (PPO)**
**選定理由:**  
- サンプル効率が高く、安定した学習が可能。
- 連続制御タスクで実績が多く、BipedalWalkerHardcoreでも多くの成功例が報告されています。
- ハイパーパラメータの調整が比較的容易で、実装も豊富。

---

### 2. **Soft Actor-Critic (SAC)**
**選定理由:**  
- 最大エントロピー強化学習により、探索性が高く、多様な状況での学習が安定。
- サンプル効率が高く、ノイズの多い環境でもロバストに学習できる。
- 連続行動空間のタスクで最先端の性能を発揮している。

---

### 3. **Twin Delayed DDPG (TD3)**
**選定理由:**  
- DDPGの改良版で、過推定バイアスを抑制し、より安定した学習が可能。
- 連続制御タスクでPPOやSACと並ぶ高い性能を示している。
- ノイズ耐性が高く、BipedalWalkerHardcoreのような難易度の高い環境にも適している。

---

### 4. **Deep Deterministic Policy Gradient (DDPG)**
**選定理由:**  
- 連続行動空間に特化したアクター・クリティック型アルゴリズム。
- 探索ノイズの導入や経験再生バッファによる効率的な学習が可能。
- TD3の登場以降はやや古典的だが、十分な性能を発揮できる。

---

### 5. **Distributed Distributional DDPG (D4PG)**
**選定理由:**  
- DDPGを分散並列化し、分布的価値推定を導入した発展型。
- 大規模並列学習と分布的強化学習による高い性能。
- より複雑な環境や大規模なタスクにも適用可能。

今回TD3を学習したことがないのでTD3を実装する。

## TD3とは
TD3（Twin Delayed Deep Deterministic Policy Gradient）は、連続行動空間を持つ環境における強化学習のためのオフポリシー型アクター・クリティックアルゴリズムです。2018年に発表され、DDPG（Deep Deterministic Policy Gradient）の改良版として知られています。

### TD3の特徴
#### 1. 過大評価バイアスの抑制
DDPGでは、Q関数（価値関数）の推定が過大になる傾向（過大評価バイアス）があり、学習が不安定になることがありました。TD3では、2つの独立したQネットワーク（クリティック）を用意し、2つのQ値のうち小さい方をターゲット値として使用することで、過大評価を抑制します。

#### 2. ターゲットポリシーのスムージング
ターゲット値を計算する際、ターゲットアクションに小さなノイズを加えてスムージングします。これにより、値関数の過学習や過剰な鋭敏さを防ぎます。

#### 3. 遅延更新（Delayed Policy Updates）
アクターネットワーク（ポリシー）の更新頻度をクリティックよりも遅くします。これにより、より安定した価値推定のもとでポリシーが更新され、学習が安定します。

### TD3の基本的な流れ
1. 2つのクリティックネットワークでQ値を計算。
2. ターゲットアクションにノイズを加えてスムージング。
3. ターゲットQ値は2つのクリティックの小さい方を利用。
4. アクター（ポリシー）ネットワークは遅延して更新。


![alt text](image-1.png)

https://medium.com/@joanna.z.gryczka/td3-tutorial-and-implementation-682f16b56699

![alt text](image-2.png)


1. The algorithm also uses the Replay Buffer, a data structure where past experiences are stored, allowing the agent to learn from its history.
It has various important advantages: increased speed of learning, reduced correlation between experiences, and reduced chance of catastrophic forgetting by reusing past experiences.
2. TD3 is model-free, which means that the agent learns the best policy directly from the interactions with environment. Consequently, it does not require knowledge of environment’s dynamics.

![alt text](image-3.png)
