強化学習におけるロスは通常のニューラルネットワークと異なりわかりづらい印象です。
少し整理してみようと思います。

強化学習における「ロス」は、教師あり学習のように一意の形があるわけではなく、 **「何を最適化したいか（価値・方策・両方）」によって定義が変わる** という点が最初の重要ポイントです。

<img src="images/hammering_pinokio.png" alt="jssp-3" width="600px" height="auto">

## 1. 強化学習におけるロスの考え方（全体像）

### 教師あり学習との違い
まずはここが一番重要なポイントです。

* 教師あり学習
  → 正解ラベルがあり、
  `予測 − 正解` の誤差を最小化する

* 強化学習
  → 正解行動は存在しない
  → **将来の報酬が最大になるように振る舞いを学習する**

そのため強化学習では、

> **「報酬を最大化する」≒「ロスを最小化する」**

ようにロスを設計します。


## 2. 価値ベース手法のロス（Q-learning 系）

### 目的

状態 `s` で行動 `a` を選んだときの
**将来報酬の期待値（Q値）** を正しく近似したい。

### 基本式（TD誤差）

$$
\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

### ロス関数（MSE）

$$
\mathcal{L}_{Q} = \mathbb{E}[\delta^2]
$$

### 直感的な意味

* 「今のQ予測」と
* 「1ステップ先の予測＋報酬」
  のズレを小さくする

### 実装イメージ（PyTorch）

```python
loss = ((reward + gamma * next_q - q_value) ** 2).mean()
```


## 3. 方策勾配法のロス（Policy Gradient）

### 目的

「良い結果をもたらした行動」を **より選びやすくする確率分布（方策）** を学習する。

### 基本式

$$
\mathcal{L}_{policy} = - \mathbb{E}[\log \pi(a|s) \cdot R]
$$

### ポイント

* 報酬が大きい行動 → 確率を上げる
* 報酬が小さい行動 → 確率を下げる

### なぜマイナス？

PyTorch は「最小化」しかできないため、
**最大化したい目的関数に − を付ける**

### 実装イメージ

```python
loss = - (log_prob * reward).mean()
```


## 4. 分散を下げる工夫（ベースライン / Advantage）

### 問題

生の報酬を使うと学習が不安定。

### 解決策

平均との差分（Advantage）を使う。

$$
A(s,a) = R - V(s)
$$

### ロス

$$
\mathcal{L}_{policy} = - \log \pi(a|s) \cdot A(s,a)
$$

### 効果

* 学習が安定
* 収束が速くなる

## 5. Actor-Critic のロス（実用で最重要）

### 構成

* Actor（方策）
* Critic（価値）

### ロスは **足し算**

$$
\mathcal{L} = \mathcal{L}_{policy} + \lambda \mathcal{L}_{value}
$$

#### 方策ロス

```python
policy_loss = -(log_prob * advantage).mean()
```

#### 価値ロス

```python
value_loss = ((V_pred - R) ** 2).mean()
```

#### 合成

```python
loss = policy_loss + 0.5 * value_loss
```


## 6. エントロピー正則化（探索促進）

### 目的

* 行動が固定化しすぎるのを防ぐ

### 追加ロス

$$
\mathcal{L}_{entropy} = -\beta H(\pi)
$$

### 実装

```python
loss = policy_loss + value_loss - beta * entropy
```


## 7. まとめ（体系表）

| 手法              | ロスの意味           |
| --------------- | --------------- |
| Q-learning      | 予測Q値とTDターゲットのズレ |
| Policy Gradient | 良い行動の確率を上げる     |
| Actor-Critic    | 方策 + 価値の同時最適化   |
| Entropy         | 探索を維持           |


## 結論
バッと説明した内容ですが、以下の通り結論とします。

* 強化学習のロスは「誤差」ではない
* **目的関数（報酬最大化）を数学的に書き換えたもの**
* 手法によってロスの定義が変わる



