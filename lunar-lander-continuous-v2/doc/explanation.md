## 環境について
Box2D

Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
LunarLander-v2 defines "solving" as getting average reward of 200 over 100 consecutive trials.=
by @olegklimov


https://elegantrl.readthedocs.io/en/latest/tutorial/LunarLanderContinuous-v2.html


how to use
```
pip install gym[box2d]
```

なんか自己証明書の検証に失敗する・・・
```
pip install gym[box2d] --trusted-host pypi.org --trusted-host files.pythonhosted.org
```


このエラーの主な原因は、`gym[box2d]`をインストールする際に**依存パッケージの`pygame==2.1.0`のインストールが失敗している**ことです。  
エラーの詳細（`python setup.py egg_info did not run successfully`）は、一般的に以下のような理由で発生します。

---

## 主な原因

### 1. **Pythonバージョンとpygameの互換性**
- `pygame==2.1.0`は、Python 3.12では**公式にサポートされていません**。
- Python 3.12対応の`pygame`は、より新しいバージョン（例：`pygame>=2.3.0`）が必要です。

### 2. **ビルド環境の不足**
- `pygame`をソースからビルドする場合、Cコンパイラや依存ライブラリ（SDL2など）が必要です。これらが不足しているとビルドエラーになります。

### 3. **依存パッケージの競合や不整合**
- 既存のパッケージと依存関係が競合している場合、インストールに失敗することがあります。

---

## 解決策

### 1. **Python 3.12で`pygame==2.1.0`は非対応なのでバージョンを合わせる**
- **推奨**: Python 3.10や3.11など、`pygame==2.1.0`がサポートされているバージョンで再度インストールする。
- または、`gym[box2d]`の依存パッケージを手動で新しいバージョンに置き換える。

### 2. **`pygame`の新しいバージョンを使う**
- もしPython 3.12を使い続けたい場合は、`pygame`の最新版（例：`2.5.2`など）をインストールする。

```bash
pip install pygame --upgrade
```

- その後、`gym[box2d]`を`--no-deps`オプション付きでインストールし、依存関係の競合を避ける。

```bash
pip install gym[box2d] --no-deps
```

### 3. **ビルド環境を整える**
- Windowsの場合は、[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)をインストールしてください。

---

## まとめ

- **原因**: Python 3.12と`pygame==2.1.0`の非互換
- **対策**:
  1. Pythonバージョンを下げる（3.10などを推奨）
  2. `pygame`の新しいバージョンを先にインストールしてから`gym[box2d]`を`--no-deps`で入れる
  3. ビルドツールが必要な場合はインストールする

---

### 参考コマンド（Python 3.12のまま進める場合）

gymnasiumで勧めること。
以下で正常稼働確認

```python
import gymnasium as gym

# 環境の作成
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
# env = gym.make("CarRacing-v2", domain_randomize=True)
# 環境の初期化
observation, info = env.reset()

for step in range(500):  # 500ステップだけ実行
    # ランダムなアクションを選択
    action = env.action_space.sample()
    # アクションを実行
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

| 項目              | 内容                                                                                                   |
|-------------------|--------------------------------------------------------------------------------------------------------|
| Action Space      | Discrete(4)                                                                                            |
| Observation Space | Box([ -2.5, -2.5, -10., -10., -6.2831855, -10., -0., -0. ], [ 2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1. ], (8,), float32) |
| import            | gymnasium.make("LunarLander-v3")                                                                       |

#### Action Space
There are four discrete actions available:
- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## エージェント設計
制御が連続なため、連続制御に適した強化学習法を選定したい。

| アルゴリズム名 | 特徴・説明 |
|---|---|
| **DDPG (Deep Deterministic Policy Gradient)** | 連続アクション空間に対応したActor-Critic型の手法。オフポリシー。 |
| **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** | DDPGの改良版。より安定で高精度。オフポリシー。 |
| **SAC (Soft Actor-Critic)** | エントロピー正則化を導入し、探索性と安定性が高い。近年人気。オフポリシー。 |
| **PPO (Proximal Policy Optimization)** | オンポリシー。連続・離散どちらも対応。安定しやすい。 |


## 実装のポイント
1. actor-cliticモデル
2. ノイズ発生することでノイズを混ぜて学習させる。これはDDPGが決定論的な方策出力方式のため、探索ができないため。探索アクションに意図的にノイズを混ぜて学習させる。
3. 行動価値関数は遅延学習させたい。このため、stepごとに重みを同期させる方法を用いる。



