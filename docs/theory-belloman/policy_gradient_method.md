
方策勾配は強化学習において、[方策を用いたモデル](https://yoshishinnze.hatenablog.com/entry/2026/08/03/043000)におけるロス関数計算の段階で非常に重要な概念です。
これがないと、[方策ベースの深層強化学習]でパラメータ更新の難易度がえらく上がってしまうことになります。

[方策ベースのパラメータ更新](https://yoshishinnze.hatenablog.com/entry/2026/09/01/043000)が現実的に解ける形に持って行ったのが方策勾配法です。

本日は方策勾配法と方策勾配法のベースとなる方策勾配定理について扱っていきます。

## 方策勾配法概要

方策勾配法（Policy Gradient Method）は、強化学習においてエージェントの行動ルール（方策）を直接最適化する手法です。価値関数を介さず、方策そのものをパラメータ化して学習する点が大きな特徴となります。

### 基本的な考え方

強化学習のアプローチには大きく分けて「価値ベース」と「方策ベース」の2つがあります。Q学習やDQNなどの価値ベースの手法では、状態や行動の価値（Q値など）を学習し、それに基づいて間接的に方策を決定します。一方、方策勾配法では方策 $\pi_\theta(a|s)$ をパラメータ $\theta$ を用いた関数として直接表現し、期待報酬が最大になるように $\theta$ を勾配法で更新します。[ゼロから作るDeep Learning ❹ ―強化学習編](https://www.oreilly.com/library/view/zerokarazuo-rudeep-learning/9784873119755/ch09.xhtml)

具体的には、以下のようにパラメータを更新します。

$$\theta \leftarrow \theta + \alpha \frac{\partial \rho}{\partial \theta}$$

ここで $\alpha$ は学習率、$\rho$ は期待収益を表します。[RLTech Lab](https://rltechlab.com/%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92%e3%81%ae%e5%9f%ba%e7%a4%8e%e2%91%a2%ef%bc%88policy%e3%83%99%e3%83%bc%e3%82%b9%e3%81%ae%e6%89%8b%e6%b3%95%ef%bc%89/)

### 方策勾配定理

方策勾配法の理論的な柱となるのが本日の話の中心である「方策勾配定理」です。これにより、期待報酬の勾配を以下の形で表現できます。

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} G(\tau) \nabla_{\theta} \log{\pi_{\theta}(A_t|S_t)} \right]$$

この式は、「良い行動（報酬が高い）を選んだ確率を増やし、悪い行動を選んだ確率を減らす」という直感に対応しています。対数確率の勾配 $\nabla_{\theta} \log \pi_{\theta}(A_t|S_t)$ に報酬 $G(\tau)$ を重みとして掛け合わせることで、報酬に応じた方策の更新が可能になります。[あつまれ統計の森](https://www.hello-statisticians.com/ml/rl/policy_gradient1)

### 代表的なアルゴリズム

方策勾配法の最も基本的なアルゴリズムは **REINFORCE** です。これはモンテカルロ法を用いてエピソード全体の報酬から勾配を推定し、方策を更新する手法です。[Qiita - 方策ベースアルゴリズムの基礎](https://qiita.com/pocokhc/items/d1cc041f04f8ad23156e)

### メリットと特徴

方策勾配法には以下のような利点があります。

- **連続行動空間への対応**: 行動が連続値である問題でも、確率分布（例えば正規分布）からのサンプリングとして方策を表現できるため、自然に扱えます。
- **確率的な行動選択**: 探索と活用のトレードオフを確率分布として表現でき、価値ベースの手法のように $\arg\max$ による決定的な選択に縛られません。
- **方策の直接最適化**: 価値関数の推定誤差が方策に伝播しないため、学習が安定する場合があります。[Chaos and Order - Policy Gradient](https://www.youngju.dev/blog/deep-rl/09_policy_gradients.ja)

一方で、モンテカルロ法によるバリアンスが大きくなりやすい、学習が遅い、といった課題もあります。これを改善するために、ベースラインの導入（Actor-Critic など）が行われることも多いです。

## 方策勾配定理

方策勾配定理の証明を、無限ホライズン・割引報酬の設定を中心に説明します。

### 前提設定

方策 $\pi_\theta(a|s)$ をパラメータ $\theta$ で表現し、**割引報酬の累積和**を最大化することを考えます。

- 状態遷移確率: $p(s'|s,a)$
- 即時報酬: $r(s,a)$（確率的報酬でも同様に成立します）
- 割引率: $\gamma \in [0,1)$
- 行動価値関数: $Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r(S_t, A_t) \,\middle|\, S_0=s, A_0=a\right]$
- 状態価値関数: $V^\pi(s) = \sum_a \pi_\theta(a|s) Q^\pi(s,a)$
- **割引状態訪問分布**: $d^\pi(s) = \sum_{t=0}^{\infty} \gamma^t P(S_t=s \mid S_0 \sim \mu, \pi)$

ここで $\theta$ は方策（Policy）を定義するパラメータ（重み）の集合で、例えばDQNの中では方策を出力するニューラルネットワークそのものを意味します。

---

__上式の導出__

目的関数 $J(\theta)$ の2つの異なる表現が等しいことを示したものです。

$$J(\theta) = \sum_s \mu(s) V^{\pi_\theta}(s) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) r(s,a)$$

これを導くために、**割引状態訪問分布**を定義します。

__定義：割引状態訪問分布__

初期状態 $S_0$ が分布 $\mu$ からサンプリングされ、方策 $\pi_\theta$ に従って行動するとき、時刻 $t$ に状態 $s$ にいる確率を $P(S_t=s \mid S_0 \sim \mu, \pi_\theta)$ と書きます。このとき、**割引状態訪問分布**を以下で定義します。

$$d^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t P(S_t=s \mid S_0 \sim \mu, \pi_\theta)$$

これは「初期状態から discounted な確率質量」として解釈できます。

__導出__

__左辺の展開__

状態価値関数の定義より：

$$V^{\pi_\theta}(s) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r(S_t, A_t) \,\middle|\, S_0=s\right]$$

これを時刻ごとに展開すると：

$$V^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t \sum_{s_t, a_t} P(S_t=s_t, A_t=a_t \mid S_0=s, \pi_\theta) \, r(s_t, a_t)$$

初期状態分布 $\mu$ で重み付けして合計します。

$$\sum_s \mu(s) V^{\pi_\theta}(s) = \sum_s \mu(s) \sum_{t=0}^{\infty} \gamma^t \sum_{s_t, a_t} P(S_t=s_t, A_t=a_t \mid S_0=s, \pi_\theta) \, r(s_t, a_t)$$

__期待値の順序を入れ替える__

和の順序を $\sum_s \mu(s) \sum_{t=0}^{\infty} \sum_{s_t, a_t}$ から $\sum_{t=0}^{\infty} \sum_{s_t, a_t} r(s_t, a_t) \sum_s \mu(s) P(\cdots)$ に入れ替えます。

$$= \sum_{t=0}^{\infty} \gamma^t \sum_{s_t, a_t} r(s_t, a_t) \underbrace{\sum_s \mu(s) \, P(S_t=s_t, A_t=a_t \mid S_0=s, \pi_\theta)}_{= P(S_t=s_t, A_t=a_t \mid S_0 \sim \mu, \pi_\theta)}$$

ここで、$\sum_s \mu(s) P(\cdots \mid S_0=s)$ は「初期状態を $\mu$ で平均化したときの同時確率」となります。これを $P(S_t=s_t, A_t=a_t \mid S_0 \sim \mu, \pi_\theta)$ と書きます。

__方策の分解__

$$P(S_t=s_t, A_t=a_t \mid S_0 \sim \mu, \pi_\theta) = P(S_t=s_t \mid S_0 \sim \mu, \pi_\theta) \cdot \pi_\theta(a_t|s_t)$$

したがって：

$$\sum_s \mu(s) V^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t \sum_{s_t} P(S_t=s_t \mid S_0 \sim \mu, \pi_\theta) \sum_{a_t} \pi_\theta(a_t|s_t) \, r(s_t, a_t)$$

__割引状態訪問分布を適用__

$\sum_{t=0}^{\infty} \gamma^t P(S_t=s_t \mid \cdots)$ の部分は、定義より $d^{\pi_\theta}(s_t)$ そのものです。したがって：

$$\boxed{\sum_s \mu(s) V^{\pi_\theta}(s) = \sum_{s_t} d^{\pi_\theta}(s_t) \sum_{a_t} \pi_\theta(a_t|s_t) \, r(s_t, a_t)}$$

これが求める等式です。

---

ということで目的関数の数式は以下のように整理されます。

ここで $\mu$ は初期状態分布です。$d^\pi(s)$ は「初期状態から discounted な確率質量」として解釈できます。

目的関数は以下で定義されます。

$$J(\theta) = \sum_s \mu(s) V^{\pi_\theta}(s) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) r(s,a)$$

より簡潔に：

$$J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) Q^{\pi_\theta}(s,a)$$

### 方策勾配定理

上記の設定のもとで、以下が成立します。

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a)$$

これを**方策勾配定理**（Policy Gradient Theorem）と呼びます。[Qiita - 方策勾配定理のすっきりした証明](https://qiita.com/itok_msi/items/3c278e06cf2b0bb32f4e) [Stanford - Policy Gradient Algorithms](https://stanford.edu/~ashlearn/RLForFinanceBook/PolicyGradient.pdf)

### 証明

__ステップ1: 状態価値関数の勾配を展開する__

まず、$V^\pi(s) = \sum_a \pi_\theta(a|s) Q^\pi(s,a)$ の両辺を $\theta$ で微分します。

$$\nabla_\theta V^\pi(s) = \sum_a \left[ \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a) + \pi_\theta(a|s) \, \nabla_\theta Q^\pi(s,a) \right] \tag{1}$$

__ステップ2: $\nabla_\theta Q^\pi(s,a)$ を展開する__

$Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} p(s'|s,a) V^\pi(s')$ です。ここで $r(s,a)$ と $p(s'|s,a)$ は環境の定数なので、$\theta$ に依存しません。したがって：

$$\nabla_\theta Q^\pi(s,a) = \gamma \sum_{s'} p(s'|s,a) \, \nabla_\theta V^\pi(s') \tag{2}$$

__ステップ3: (2) を (1) に代入する__

(1) に (2) を代入すると：

$$\nabla_\theta V^\pi(s) = \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a) + \sum_a \pi_\theta(a|s) \left[ \gamma \sum_{s'} p(s'|s,a) \, \nabla_\theta V^\pi(s') \right]$$

第2項を整理すると：

$$\nabla_\theta V^\pi(s) = \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a) + \gamma \sum_{s'} \left( \sum_a \pi_\theta(a|s) p(s'|s,a) \right) \nabla_\theta V^\pi(s')$$

ここで $\sum_a \pi_\theta(a|s) p(s'|s,a) = P(S_{t+1}=s' \mid S_t=s, \pi)$ は、方策 $\pi$ のもとでの状態遷移確率です。これを $P^\pi(s \to s')$ と書くことにします。

$$\nabla_\theta V^\pi(s) = \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a) + \gamma \sum_{s'} P^\pi(s \to s') \, \nabla_\theta V^\pi(s') \tag{3}$$

__ステップ4: 再帰的に展開する__

(3) は $\nabla_\theta V^\pi(s)$ についての再帰式です。これを $\nabla_\theta V^\pi(s')$ に対しても同様に展開し、繰り返し代入していくと：

$$\nabla_\theta V^\pi(s) = \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a) + \gamma \sum_{s'} P^\pi(s \to s') \sum_a \nabla_\theta \pi_\theta(a|s') \, Q^\pi(s',a) + \gamma^2 \sum_{s''} P^\pi(s \to s'' \text{ in 2 steps}) \nabla_\theta V^\pi(s'') + \cdots$$

ここで $P^\pi(s \to s' \text{ in } t \text{ steps})$ は、方策 $\pi$ のもとで $t$ ステップ後に状態 $s'$ に到達する確率です。

$\gamma < 1$ なので、$t \to \infty$ で $\gamma^t \nabla_\theta V^\pi(\cdot) \to 0$ となり、最後の項は消えます。したがって：

$$\nabla_\theta V^\pi(s) = \sum_{t=0}^{\infty} \sum_{s_t} \gamma^t P^\pi(s \to s_t \text{ in } t \text{ steps}) \sum_a \nabla_\theta \pi_\theta(a|s_t) \, Q^\pi(s_t,a) \tag{4}$$

__ステップ5: 初期状態分布で重み付けして合計する__

目的関数は $J(\theta) = \sum_s \mu(s) V^\pi(s)$ なので：

$$\nabla_\theta J(\theta) = \sum_s \mu(s) \nabla_\theta V^\pi(s)$$

(4) を代入すると：

$$\nabla_\theta J(\theta) = \sum_s \mu(s) \sum_{t=0}^{\infty} \sum_{s_t} \gamma^t P^\pi(s \to s_t \text{ in } t \text{ steps}) \sum_a \nabla_\theta \pi_\theta(a|s_t) \, Q^\pi(s_t,a)$$

$s$ と $t$ の順序を入れ替え、$s_t$ を $s$ と書き直すと：

$$\nabla_\theta J(\theta) = \sum_s \left[ \sum_{t=0}^{\infty} \gamma^t P(S_t=s \mid S_0 \sim \mu, \pi) \right] \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^\pi(s,a)$$

カギ括弧内の項はまさに **割引状態訪問分布** $d^{\pi_\theta}(s)$ の定義そのものです。したがって：

$$\boxed{\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a)}$$

これが方策勾配定理です。[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

### 対数導関数による書き換え

実際のアルゴリズムでは、上式を以下のように書き換えて使用します。

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, \, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a) \right]$$

これは $\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$ という対数導関数の性質（log-derivative trick）から直ちに従います。[あつまれ統計の森](https://www.hello-statisticians.com/ml/rl/policy_gradient1)

### 直感的な意味

方策勾配定理の式は、以下の直感を数学的に表現しています。

- $\nabla_\theta \pi_\theta(a|s)$ は「方策を変えたときに、状態 $s$ で行動 $a$ を選ぶ確率がどう変化するか」を表します。
- $Q^{\pi_\theta}(s,a)$ は「その行動 $a$ が長期的にどれだけ良いか」を表します。
- したがって、$Q$ 値が大きい行動の確率を増やし、$Q$ 値が小さい行動の確率を減らす方向に勾配が向きます。

この構造により、価値関数を介さずに方策を直接最適化できることが保証されます。

## コードに実装すると

方策勾配法の代表的なアルゴリズムである **REINFORCE** を PyTorch で実装する場合のコード構成を、CartPole を例に説明します。

### REINFORCE アルゴリズムの流れ

1. エピソードを1つ完了させ、状態・行動・報酬の系列を記録する
2. 各時刻の**割引累積報酬（リターン）** $G_t$ を計算する
3. 損失関数 $\text{Loss} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$ を計算する
4. 勾配を求めてパラメータ $\theta$ を更新する
5. 上記を繰り返す

### PyTorch 実装コード

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym  # または import gym


# ============================
# 1. 方策ネットワークの定義
# ============================
class PolicyNetwork(nn.Module):
    """
    状態 s を入力として、各行動 a の選択確率を出力するネットワーク。
    このネットワークの重み・バイアスが「θ」に相当する。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)  # 未正規化のスコア
        return logits

    def select_action(self, state):
        """
        状態から行動を確率的に選択し、その対数確率も返す。
        """
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)       # π_θ(a|s)
        dist = Categorical(probs)                    # カテゴリカル分布
        action = dist.sample()                      # 行動をサンプリング
        log_prob = dist.log_prob(action)             # log π_θ(a|s)
        return action.item(), log_prob


# ============================
# 2. 割引累積報酬（リターン）の計算
# ============================
def compute_returns(rewards, gamma):
    """
    rewards: 1エピソードの即時報酬リスト [r_0, r_1, ..., r_T]
    gamma  : 割引率
    戻り値 : G_t のリスト（各時刻の割引累積報酬）
    """
    returns = []
    G = 0
    # エピソード末尾から逆順に計算
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # 平均0・分散1に正規化（学習の安定化）
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


# ============================
# 3. メイン学習ループ
# ============================
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    gamma = 0.99
    num_episodes = 1000

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []   # log π_θ(a_t|s_t) を保存
        rewards = []     # r_t を保存
        done = False

        # ---- 1エピソードの実行 ----
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = policy.select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # ---- リターンの計算 ----
        returns = compute_returns(rewards, gamma)

        # ---- 損失関数の計算 ----
        # Loss = - Σ_t log π_θ(a_t|s_t) * G_t
        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss -= log_prob * Gt

        # ---- 勾配更新 ----
        optimizer.zero_grad()
        loss.backward()   # ∇_θ J(θ) を計算
        optimizer.step()  # θ ← θ + α ∇_θ J(θ)

        # 進捗表示
        total_reward = sum(rewards)
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    train()
```

### コードのポイント解説

| 部分 | 数式との対応 | 説明 |
|------|-------------|------|
| `PolicyNetwork` | $\pi_\theta(a\|s)$ | パラメータ $\theta$ を持つ方策をニューラルネットワークで表現 |
| `select_action` | $a \sim \pi_\theta(\cdot\|s)$ | 方策から確率的に行動をサンプリング |
| `dist.log_prob(action)` | $\log \pi_\theta(a\|s)$ | 選択した行動の対数確率を取得 |
| `compute_returns` | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ | 各時刻の割引累積報酬を計算 |
| `loss -= log_prob * Gt` | $-\log \pi_\theta(a\|s) \cdot G_t$ | 方策勾配定理に基づく損失関数 |
| `loss.backward()` | $\nabla_\theta J(\theta)$ | PyTorch が自動で勾配を計算 |
| `optimizer.step()` | $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ | 勾配方向にパラメータを更新 |

### 損失関数の符号について

通常の機械学習では損失関数を**最小化**しますが、方策勾配法では期待報酬を**最大化**したいため、損失関数として $-\log \pi_\theta(a|s) \cdot G_t$ を使います。

- $G_t > 0$（良い行動）のとき: 損失を減らす方向に学習するため、その行動の確率が増加
- $G_t < 0$（悪い行動）のとき: 損失を増やす方向に学習するため、その行動の確率が減少

### 参考文献

- [GitHub - REINFORCE CartPole PyTorch](https://github.com/ProfessorDong/Deep-Learning-Course-Examples/blob/master/DRL_Examples/REINFORCE_CartPole_PyTorch.py)
- [Zenn - 強化学習をPyTorchで実装 方策勾配法編](https://zenn.dev/takesan150/articles/5e5e86638f4c3d)
- [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

## 総括

### 方策勾配法の本質

強化学習において、エージェントの行動ルール（方策）を**価値関数を介さず直接最適化する手法**です。方策 $\pi_\theta(a|s)$ をニューラルネットワークなどのパラメータ $\theta$ で表現し、期待報酬 $J(\theta)$ を最大化する方向に勾配法で $\theta$ を更新します。

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$


### 方策勾配定理の本質

期待報酬の勾配を、以下の形で表現できることを示した定理です。

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a)$$

これを対数導関数で書き換えると：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, \, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s,a) \right]$$

**核心の直感**: 「良い行動（$Q$ 値が高い）を選んだ確率を増やし、悪い行動を選んだ確率を減らす」。対数確率の勾配に $Q$ 値を重みとして掛けることで、報酬に応じた方策の更新が可能になります。


### まとめ

方策勾配法とは、**方策をパラメータ化して勾配法で直接学習する**強化学習のアプローチであり、その理論的基盤となる方策勾配定理により、**期待報酬の勾配を方策の対数確率と行動価値の積の期待値として計算できる**ことが保証されます。これにより、連続行動空間への対応や確率的な行動選択が自然に実現でき、深層強化学習において Actor-Critic などの発展的手法の基礎となっています。
