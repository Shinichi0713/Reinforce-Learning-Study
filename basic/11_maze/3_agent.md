[昨日の環境コード](https://yoshishinnze.hatenablog.com/entry/2025/10/29/000000)をもとに強化学習するエージェントの構築を行いたいと思います。
説明の流れは強化学習のエージェントが学習する仕組みの説明→実装という流れで進めていきます。

本日テーマ：

> 強化学習のエージェントの仕組みの説明と実装

## 強化学習エージェントが学習する仕組み

強化学習エージェントが学習する仕組みは、大きく分けて以下の流れになります。

### 1. 全体の流れ（エージェントの学習サイクル）

1. **環境から状態を受け取る**

   - 例：迷路の現在位置 `(x, y)`。
2. **状態に基づいて行動を選ぶ**

   - 例：上・下・左・右のどれかを選ぶ。
3. **環境に行動を送り、結果を受け取る**

   - `step(action)` を呼び出し、
     - 次の状態 `next_state`
     - 報酬 `reward`
     - 終了フラグ `done`
       を受け取る。
4. **受け取った結果を使って「価値」を更新する**

   - 「この状態でこの行動を選ぶと、将来どれだけ報酬が得られるか」という**価値（Q値など）**を更新。
5. **状態を更新し、終了まで繰り返す**

   - `state = next_state` として、ゴールに着くまで 2〜4 を繰り返す。
6. **エピソードをリセットして再開**

   - `reset()` で初期状態に戻し、何度も繰り返して学習する。

### 2. 行動の選び方（ポリシー）

エージェントは、**「どの行動を選ぶか」を決めるルール（ポリシー）** を持っています。

__(1) 探索と活用のバランス__

- **活用（Exploitation）**: 今まで学んだ中で「一番良さそうな行動」を選ぶ。
- **探索（Exploration）**: まだ試していない行動や、たまにはランダムな行動を選ぶ。

例：ε-greedy法

- 確率 ε でランダム行動（探索）
- 確率 1-ε で最適と思われる行動（活用）

__(2) 価値に基づいて行動を選ぶ__

- 各状態・行動の組み合わせに対して「価値（Q値）」を保持。
- 例：`Q[(0,0), 右] = 5.0` なら、「(0,0)で右に行くのは価値が高い」と判断。

### 3. 価値の更新ロジック（例：Q学習）

__目的__

- 「状態 s で行動 a を選んだときの、将来の報酬合計の期待値」を学習する。

__Q学習の更新式（イメージ）__

```
Q[s, a] ← Q[s, a] + α * (reward + γ * max_a' Q[next_state, a'] - Q[s, a])
```

- `α`（学習率）: どれだけ急に更新するか
- `γ`（割引率）: 将来の報酬をどれだけ重視するか
- `max_a' Q[next_state, a']`: 次の状態で取りうる最良の行動の価値

__直感的な意味__

- 「実際に得られた報酬 + 将来の見込み報酬」と「今の予測値」の差を埋めるように更新する。
- これを繰り返すことで、Q値が「真の価値」に近づいていく。

### 4. 迷路の例で見る具体的な学習ロジック

1. 初期状態: `state = (0,0)`（スタート）
2. 行動選択: ε-greedy で行動を選ぶ（例：右）
3. `step(右)` を呼ぶ →
   - `next_state = (0,1)`
   - `reward = 0`
   - `done = False`
4. Q値を更新:
   - 「(0,0)で右に行ったときの価値」を、`reward=0` と `(0,1)` での最良行動の価値を使って更新。
5. `state = (0,1)` に更新し、次のステップへ。
6. ゴールに着くまで繰り返す。
7. 何度もエピソードを繰り返すうちに、
   - ゴールに近づく行動の Q値が高くなり、
   - 壁にぶつかる行動の Q値が低くなる。

### 5. 学習が進むとどうなるか

- エージェントは「どの状態でどの行動を選ぶと、将来の報酬が最大になるか」を学習します。
- 迷路の例では：
  - 壁を避けつつ
  - ゴールに最短で到達する経路
    を選ぶようになります。

## 実装

以下に、強化学習のコード実装におけるキーポイントを整理しつつ、迷路環境に対する Q学習の実装例を交えて説明します。

### 1. 強化学習コード実装のキーポイント（全体像）

__(1) 環境インターフェースの統一__

- `reset()`: 環境を初期状態に戻し、初期状態を返す
- `step(action)`: 行動を実行し、`(next_state, reward, done)` を返す
- これにより、アルゴリズム側は環境に依存せず実装できます。

__(2) エージェントの基本構造__

- 状態・行動から価値を表すデータ構造（例：Qテーブル）を持つ
- 行動選択ポリシー（例：ε-greedy）を実装
- 価値更新ロジック（例：Q学習の更新式）を実装

__(3) 学習ループの設計__

- エピソードループ（1ゲーム分の学習）
- ステップループ（1ステップごとの行動・更新）
- 探索と活用のバランス（εの調整）

__(4) ハイパーパラメータの設定__

- 学習率 α
- 割引率 γ
- 探索率 ε
- エピソード数

### 2. 迷路環境に対する Q学習の実装例（Python）

__前提__

- 前回までの `MazeEnv` クラスが利用可能とします。
- 状態は `(x, y)` のタプル、行動は 0〜3（上・下・左・右）とします。

__コード例__

```python
import numpy as np
from maze_env import MazeEnv  # 前回の環境クラス

class QLearningAgent:
    """
    Q学習エージェント（迷路用）
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # 探索率

        # Qテーブルの初期化（状態: (x,y), 行動: 0~3）
        self.q_table = {}

    def get_q_value(self, state, action):
        """状態・行動に対するQ値を取得（未登録なら0で初期化）"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)  # 4行動分
        return self.q_table[state][action]

    def choose_action(self, state):
        """ε-greedy法で行動を選択"""
        if np.random.random() < self.epsilon:
            # 探索: ランダム行動
            return np.random.randint(4)
        else:
            # 活用: Q値が最大の行動を選択
            q_values = [self.get_q_value(state, a) for a in range(4)]
            return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state):
        """Q値を更新（Q学習の更新式）"""
        current_q = self.get_q_value(state, action)
        # 次の状態での最大Q値
        next_max_q = max([self.get_q_value(next_state, a) for a in range(4)])
        # Q学習の更新式
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def train(self, episodes=1000):
        """学習ループ"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

    def play(self, max_steps=50):
        """学習済みポリシーで1エピソードプレイ"""
        state = self.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = np.argmax([self.get_q_value(state, a) for a in range(4)])
            next_state, reward, done = self.env.step(action)
            self.env.render()  # 可視化
            state = next_state
            steps += 1

# --- 使用例 ---
if __name__ == "__main__":
    env = MazeEnv("maze.txt")
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    agent.train(episodes=500)
    print("Training finished. Playing with learned policy:")
    agent.play()
```

### 3. 実装のキーポイント（詳細）

__(1) Qテーブルの設計__

- キー: 状態（例：`(x, y)`）
- 値: 各行動に対するQ値の配列（例：`[上, 下, 左, 右]`）
- 未登録の状態は0で初期化（`get_q_value` 内で動的に作成）

__(2) 行動選択（ε-greedy）__

- 確率 ε でランダム行動 → 探索
- 確率 1-ε でQ値最大の行動 → 活用
- ε は最初は大きめ（0.1〜0.3）、学習が進むにつれて小さくするのも有効です。

__(3) Q値の更新（Q学習）__

- 更新式:`Q[s,a] ← Q[s,a] + α * (r + γ * max_a' Q[s',a'] - Q[s,a])`
- `r + γ * max_a' Q[s',a']` が「実際の見込み報酬」
- 現在の予測 `Q[s,a]` との差を学習率 α で埋める

__(4) 学習ループの構造__

- 外側ループ: エピソード数（何回ゲームをプレイするか）
- 内側ループ: 1エピソード内のステップ（行動・更新の繰り返し）
- 各エピソード終了時に `env.reset()` を呼び、初期状態に戻す

__(5) ハイパーパラメータの役割__

- **α（学習率）**: 大きいほど急に更新（0.1前後が一般的）
- **γ（割引率）**: 将来の報酬をどれだけ重視するか（0.9〜0.99）
- **ε（探索率）**: 探索と活用のバランス（0.05〜0.3）

### 4. 学習が進むとどうなるか（迷路の例）

そうなるだろう、という期待を込めてですが。

- 初期: Qテーブルはほぼ0 → ランダム行動
- 学習後:
  - ゴールに近づく行動のQ値が高くなる
  - 壁にぶつかる行動のQ値が低くなる
  - 結果として、最短経路でゴールに到達するようになる

## 学習の結果

今回の強化学習モデルがうまく進んでいるかは、学習の回数(Episode)が進むにつれて `Reward`の値が大きくなっているかということです。
今回モデルで実際に学習させた結果を見てみましょう。

```
Episode 0, Total Reward: -26
Episode 100, Total Reward: 9
Episode 200, Total Reward: 10
Episode 300, Total Reward: 9
Episode 400, Total Reward: 10
```

徐々に `Reward`が大きくなっていることが確認出来ます。
実際に動作させた様子はこのようになります。エージェントがゴール出来ました。

<img src="maze_demo.gif" width="500">

## 総括

強化学習エージェントは、

- **環境との相互作用（状態・行動・報酬）** を通じて、
- **価値関数（Q値など）を更新し**、
- **探索と活用のバランスを取りながら行動を選ぶ**

という仕組みで学習します。

迷路の例では、この仕組みをコードとして実装し、

- 環境クラス（`MazeEnv`）
- Q学習エージェント（`QLearningAgent`）
- 報酬設計（ゴール+10、壁-1、通常0）
- 学習ループとハイパーパラメータ調整

を組み合わせることで、エージェントが「ゴールに最短で到達する」行動を自律的に学習できることを示しました。
この枠組みは、迷路だけでなく、さまざまな強化学習タスクに応用可能です。

最期まで御覧くださってありがとうございました。
