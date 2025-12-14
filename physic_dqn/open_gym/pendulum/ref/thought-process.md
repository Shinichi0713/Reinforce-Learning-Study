## SACの実装法

SAC（Soft Actor-Critic）は、深層強化学習の代表的なオフポリシー手法のひとつです。  
最大の特徴は**エントロピー正則化**を導入して「探索性」と「安定性」を両立している点です。  
ここではSACの基本的な実装手順を、Python＋PyTorchベースで簡単に解説します。

---

### 1. 準備

- 必要なライブラリ：`torch`, `numpy`, `gym` など
- 環境：連続値行動空間（例：Pendulum-v1, Mujoco系タスクなど）


### 2. ネットワークの構成

SACでは以下のネットワークを用います。

- **ポリシーネットワーク（Actor）**：確率的ポリシー（例：ガウス分布の平均と分散を出力）
- **2つのQネットワーク（Critic）**：2つ用意し、最小値を使う（Double Q-learning）
- **値関数ネットワーク（Vネットワーク）**（省略可能：SAC-v1のみ。SAC-v2以降は不要）


### 3. 経験再生バッファの用意

- 通常のReplay Bufferを使います。


### 4. 損失関数

#### 1. Qネットワーク（Critic）の損失
```python
# y = r + gamma * (min(Q1', Q2') - alpha * log_prob(a'))
# Q1_loss = MSE(Q1(s,a), y)
# Q2_loss = MSE(Q2(s,a), y)
```

#### 2. ポリシーネットワーク（Actor）の損失
```python
# J_pi = E[alpha * log_prob(a|s) - min(Q1(s,a), Q2(s,a))]
```

#### 3. エントロピー温度パラメータ alpha の自動調整（オプション）
```python
# J_alpha = E[-alpha * (log_prob(a|s) + target_entropy)]
```


### 5. 学習の流れ

1. 環境から状態sを取得
2. ポリシーから行動aをサンプリング
3. 環境にaを与えて次状態s', 報酬r, doneを得る
4. (s, a, r, s', done)をReplay Bufferに保存
5. 一定ステップごとにバッファからミニバッチをサンプリングし、上記損失関数でネットワークを更新


### 6. 最小構成の擬似コード

```python
for each training step:
    # 1. サンプリング
    a = policy(s)
    s', r, done = env.step(a)
    replay_buffer.add(s, a, r, s', done)
    s = s'

    # 2. バッチ学習
    batch = replay_buffer.sample()
    # Qネットワーク更新
    # ポリシーネットワーク更新
    # alphaの自動調整（必要なら）
    # ターゲットネットワークの更新
```


### 7. 参考実装

- [OpenAI Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [pytorch-soft-actor-critic (GitHub)](https://github.com/pranz24/pytorch-soft-actor-critic)
- [rlkit (GitHub)](https://github.com/vitchyr/rlkit)


### ポイントまとめ

- **ポリシーは確率的（ガウス分布など）**
- **Qネットワークは2つ用意**
- **エントロピー項（α）で探索性を制御**
- **Replay Bufferでオフポリシー学習**
- **パラメータαは自動調整可能**

---

>**連続制御**  
>「エージェントが取る行動（アクション）が連続的な値（実数値）で表される制御問題」
>連続値制御の概要
>- **離散値制御**：
>行動が「右に進む」「左に進む」「止まる」など、選択肢が有限個のケース
例：囲碁、チェス、グリッドワールド
>- **連続値制御**：
行動が「-1.0〜1.0の範囲の実数値」など、無限に多くの値を取るケース
例：ロボットの関節角度・速度、車のハンドル角、加速度など

---

## Soft-Q学習からSACへ
連続値制御のための有力手法である Soft Actor-Critic (SAC) の解説と、tensorflow2での実装例です。


Soft-Q Learning論文： [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)

SAC論文 ①： [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

SAC論文 ②： [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

SAC論文 ③： https://arxiv.org/pdf/1812.11103.pdf

## 個人的な所感
1. SACは連続制御に適した手法
2. Soft-Q学習と確率的方策を導入することで、探索力にも優れた手法とした。(決定論的手法ではなく、確率的方策としたことで、探索も可能)
3. 報酬に方策エントロピー項を組み込むことで、報酬の最大化と、探索両方でバランスをとることを可能とした。
4. 従来の探索的手法は確率的なランダムを用いるものが主流。ですが、これに対して、報酬に方策エントロピを組み込むことで、報酬最大化も探索もQ学習のエージェントが、意思を持って行うということがコンセプト。
5.  ”SoftQ学習における即時報酬 ＝ 通常の即時報酬 ＋ 遷移先でのエントロピーボーナス” というように報酬が変化した。
6. エントロピーがつくことで、多様な行動を促すようになります→安定的に効率的な探索をおこなうことができるようになる。



ご提示いただいた数式にはいくつか修正点があります。

1. `frac` の前に `\` が抜けています（`\frac`）。
2. 指数関数の中身が `\frac{1}{\beta Q(s,a)}` となっていますが、通常は `\beta Q(s,a)` の形（または `Q(s,a)/\beta`）です。
3. 分母の添え字が `a_i` となるべきです。
4. LaTeXの文法的なミスを修正します。


### 一般的なBoltzmann（ソフトマックス）ポリシーの式

\[
\pi(a \mid s) = \frac{e^{\beta Q(s,a)}}{\sum_{i} e^{\beta Q(s,a_i)}}
\]

または、\(\beta\) を分母に持たせる場合は

\[
\pi(a \mid s) = \frac{e^{Q(s,a)/\beta}}{\sum_{i} e^{Q(s,a_i)/\beta}}
\]


### 修正版

#### 1. \(\beta Q(s,a)\) の形
```math
\pi(a \mid s) = \frac{e^{\beta Q(s,a)}}{\sum_{i} e^{\beta Q(s,a_i)}}
```

#### 2. \(Q(s,a)/\beta\) の形
```math
\pi(a \mid s) = \frac{e^{Q(s,a)/\beta}}{\sum_{i} e^{Q(s,a_i)/\beta}}
```

---



