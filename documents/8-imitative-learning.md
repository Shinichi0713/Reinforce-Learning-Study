# 模倣学習とは

模倣学習は、 機械学習 熟練したデモンストレーターの行動を模倣することで、エージェントにタスクの実行を教えることに重点を置いたアプローチです。

このアプローチは、明示的な報酬関数を定義することが困難または非現実的なシナリオで特に役立ちます。

https://www.ai-gakkai.or.jp/jsai2006/program/pdf/100137.pdf


強化学習における**模倣学習（Imitation Learning）**とは、「**専門家（エキスパート）の行動データを学習して、エージェントが同じような行動を取れるようにする手法**」です。  
通常の強化学習は「報酬」に基づいて最適な行動を自ら試行錯誤して学びますが、模倣学習は「**お手本となる行動データ（専門家の軌跡）」を使って、より効率的に良い方策を学ぶ**というアプローチです。

---

## 模倣学習の主な種類

### 1. 行動クローン（Behavior Cloning, BC）
- **教師あり学習**としてエキスパートの状態-行動ペアを学習
- 例：状態が入力、エキスパートの行動がラベルとなる分類問題として学習
- シンプルだが、エキスパートから外れた状態では性能が落ちやすい（分布シフト問題）

### 2. 逆強化学習（Inverse Reinforcement Learning, IRL）
- エキスパートの行動から「**どんな報酬関数であればその行動が最適か**」を推定し、その報酬関数で強化学習を行う
- 報酬設計が難しいタスクで有効

### 3. Generative Adversarial Imitation Learning（GAIL）
- GAN（敵対的生成ネットワーク）のアイデアを応用
- 「エージェントの行動とエキスパートの行動が区別できないような方策」を学習
- より頑健な模倣学習が可能

---

## 模倣学習のメリット・用途

- **報酬設計が難しいタスク**でも、エキスパートデータがあれば効率よく学習できる
- 強化学習の探索効率を大幅に向上させる（初期方策の良いスタート地点を得られる）
- 実ロボットや自動運転など「失敗が許されない」タスクで多用

---

## 典型的な流れ（行動クローンの場合）

1. エキスパート（人間や既存アルゴリズム）による状態-行動データを収集
2. ニューラルネット等で「状態→行動」を予測するモデルを教師あり学習
3. 学習済みモデルを強化学習エージェントとして利用
4. 必要に応じて強化学習で微調整（ファインチューニング）

---

## まとめ

模倣学習は、**「お手本となる行動データ」から効率よく方策を学ぶ手法**であり、  
報酬設計が困難・探索が難しい問題で特に有効です。  
強化学習と組み合わせることで、より高性能なエージェントを構築できます。



もちろんです！  
ここでは**行動クローン（Behavior Cloning, BC）**による模倣学習のシンプルな実装例（PyTorch）を紹介します。  
例として、2次元グリッド上で「エキスパートがゴールまで最短経路をたどる」行動データを使い、エージェントが同じような行動を学ぶ状況を考えます。

---

## 1. エキスパートデータの生成（例）

```python
import numpy as np

def generate_expert_data(num_samples=1000, grid_size=5):
    states = []
    actions = []
    for _ in range(num_samples):
        # ランダムなスタート、ゴール
        start = np.random.randint(0, grid_size, size=2)
        goal = np.random.randint(0, grid_size, size=2)
        state = start.copy()
        while not np.array_equal(state, goal):
            # ゴールへの最短経路（上下左右移動のみ）
            if state[0] < goal[0]:
                action = 0  # down
                state[0] += 1
            elif state[0] > goal[0]:
                action = 1  # up
                state[0] -= 1
            elif state[1] < goal[1]:
                action = 2  # right
                state[1] += 1
            else:
                action = 3  # left
                state[1] -= 1
            states.append(np.concatenate([state, goal]))
            actions.append(action)
    return np.array(states), np.array(actions)
```

- **状態**: [現在位置(x, y), ゴール位置(x, y)]
- **行動**: 0=down, 1=up, 2=right, 3=left

---

## 2. ニューラルネットワークによる方策モデル

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4つの行動
        )

    def forward(self, x):
        return self.fc(x)
```

---

## 3. 教師あり学習（行動クローン）

```python
# データ準備
states, actions = generate_expert_data(num_samples=5000)
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.long)

# モデルと最適化
model = PolicyNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 学習ループ
for epoch in range(20):
    logits = model(states)
    loss = criterion(logits, actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 4. 学習済みモデルのテスト

```python
def select_action(model, state, goal):
    input_ = torch.tensor(np.concatenate([state, goal]), dtype=torch.float32)
    with torch.no_grad():
        logits = model(input_)
        action = torch.argmax(logits).item()
    return action

# 例: ランダムなスタート・ゴールで行動選択
state = np.array([0, 0])
goal = np.array([4, 4])
for t in range(10):
    action = select_action(model, state, goal)
    print(f"Step {t}: State {state}, Action {action}")
    if action == 0:
        state[0] += 1
    elif action == 1:
        state[0] -= 1
    elif action == 2:
        state[1] += 1
    elif action == 3:
        state[1] -= 1
    if np.array_equal(state, goal):
        print("Goal reached!")
        break
```

---

## ポイント

- 状態→行動の「分類問題」として学習するのが行動クローンの特徴です。
- より複雑な環境や連続値行動にも拡張できます。

---

このように、模倣学習（行動クローン）は「エキスパートの状態・行動データを教師あり学習で再現する」だけで簡単に実装できます。


