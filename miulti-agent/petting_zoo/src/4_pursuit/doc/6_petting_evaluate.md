
マルチエージェントの強化学習(MARL)のチーム連携が必要となるゲーム[Pursuit](https://yoshishinnze.hatenablog.com/entry/2026/06/06/043000)でAIエージェント学習の実装を進めています。

実装は以下の順に沿って進めています。

1. **環境ラッパ**でPursuitをMAPPO向けに整形
2. **Actor/Criticネットワーク**を設計
3. **バッファ**で経験を保存・計算
4. **学習ループ**でPPO更新を繰り返す
5. **評価・可視化**で性能を確認

前回4. 学習ループの実装まで進めました。
今回は5. 評価・可視化の実装を進めます。

今回のテーマ：
>Pursuitの環境で学習時の性能をモニタリングする可視化・評価機能を実装


## モニタリング機能とは

**学習時の性能をモニタリングする可視化・評価機能**とは、学習中のエージェントが「どれだけうまくタスクを解けているか」を**定量的・定性的に確認する仕組み**です。
著者は指標を学習サイクルの中で出力でおかしい点がないかをモニタリングしています。
強化学習に限らず、学習と名の付く手法を用いる場合は必須と言える機能だと思います。

### 具体的にはどんな機能か？

今回のPursuiであれば以下が必要な具体例と思います。
学習の進捗を計測する指標は、大きく分けて**報酬系・タスク成功系・学習安定性系**の3つに整理できます。

__1. 報酬系の指標（最も基本的）__

__(1) エピソード報酬（Episode Reward）__
- **定義**: 1エピソードで得られた報酬の合計。
- **例**: Pursuitなら「1エピソードでどれだけ捕獲・接触・ペナルティが発生したか」の合計。
- **使い方**:
  - 学習ステップごとに**移動平均**（例: 直近100エピソードの平均）を計算し、**上昇傾向**にあるか確認。
  - ランダムエージェントの平均報酬と比較し、学習済みポリシーが明らかに優れているか確認。

__(2) ステップあたり報酬（Reward per Step）__
- **定義**: 1ステップあたりの平均報酬（総報酬 ÷ ステップ数）。
- **使い方**:
  - エージェントが「無駄な動きを減らし、効率的に行動できているか」を確認。
  - Pursuitなら「-0.1のステップペナルティをどれだけ減らせているか」を見る指標にもなります。

__2. タスク成功系の指標（Pursuit向け）__

__(1) 捕獲数・成功率__
- **定義**:
  - 1エピソードで「逃走者を完全に囲んで捕獲した回数」。
  - 成功率 = （捕獲が発生したエピソード数）÷（全エピソード数）。
- **使い方**:
  - 報酬が増えていなくても、**捕獲数が増えていれば学習は進んでいる**と判断できます。
  - Pursuitでは「囲み捕獲」が最終目標なので、**最も直感的な進捗指標**です。

__(2) エピソード長（Episode Length）__
- **定義**: 1エピソードが終了するまでのステップ数。
- **使い方**:
  - タスクが「早く終わらせるほど良い」場合（例: 迷路脱出）は、**短くなるほど良い**。
  - Pursuitでは「早く捕獲できた方が良い」ので、**エピソード長が短くなれば学習が進んでいる**と判断できます。

__(3) 接触回数（Touch Count）__
- **定義**: 1エピソードで「逃走者に触れた回数」。
- **使い方**:
  - 捕獲に至らなくても「接触回数が増えている」なら、追跡の精度が上がっている可能性があります。

__3. 学習安定性・内部状態の指標__

__(1) 損失（Loss）の推移__
- **Actor Loss（ポリシー損失）**:
  - PPOのクリップ付き目的関数の値。
  - 急激に増減する場合は、学習率が高すぎる・クリップ範囲が不適切な可能性。
- **Critic Loss（価値関数損失）**:
  - V(s) と実際のリターンのMSE。
  - 0に近づくほど、価値関数が正確に学習できている。
- **使い方**:
  - Actor/Criticの損失が**安定して減少**しているか確認。
  - 損失が振動する場合は、ハイパーパラメータ調整のサイン。

__(2) エントロピー（Entropy）__
- **定義**: 行動確率分布のエントロピー（ランダムさ）。
- **使い方**:
  - エントロピーが高すぎる → まだ探索中（学習初期）。
  - エントロピーが低すぎる → 探索が足りず、局所解に陥っている可能性。
  - 適度に減少しながら安定するのが理想的。

__(3) Advantage（アドバンテージ）の統計__
- **平均・分散**:
  - Advantageが**0付近で分散が小さくなる**と、ポリシーが安定してきているサイン。
- **使い方**:
  - Advantageが常に正 or 常に負だと、価値関数の学習が不十分な可能性。

__(4) 勾配のノルム（Gradient Norm）__
- **定義**: ネットワークパラメータの勾配のL2ノルム。
- **使い方**:
  - 勾配ノルムが急激に増減する → 学習が不安定（学習率が高すぎる等）。
  - 適度に減少しながら安定するのが望ましい。

__4. Pursuit向けの実用的な組み合わせ__

Pursuitのようなマルチエージェント追跡タスクでは、以下の組み合わせが分かりやすいです。

1. **主要指標**
   - エピソード報酬の移動平均（直近100エピソード）
   - 1エピソードあたりの捕獲数（成功率）
   - エピソード長（短くなっているか）

2. **補助指標**
   - Actor/Critic Loss の推移
   - エントロピーの推移（探索が適切か）
   - 動画での挙動確認（協調して囲めているか）



### なぜモニタリングが必要なのか？

1. **学習が進んでいるかどうかの確認**
   - 報酬が増えていなければ、ハイパーパラメータやネットワーク構造に問題がある可能性があります。
   - 例: 報酬がずっと -0.1 付近なら、エージェントが何も学習できていない。

2. **実装ミスの早期発見**
   - 観測の整形ミス、報酬の計算ミス、ネットワークの形状ミスなどは、モニタリングで気づきやすくなります。
   - 例: Criticの出力が常に0付近なら、価値関数が学習できていない可能性。

3. **ハイパーパラメータの調整**
   - 学習率・クリップ範囲・GAEのλなど、PPOのパラメータを「報酬の推移」や「動画での挙動」を見ながら調整できます。

4. **収束のタイミングの判断**
   - 報酬が頭打ちになったら学習を止める、あるいは探索を増やすなどの判断材料になります。

5. **再現性と比較のため**
   - 同じ設定で複数回学習したときの性能のばらつきを確認し、アルゴリズムの安定性を評価できます。

## エントロピとは

主要指標の中でエントロピについては少し分かりづらいので補足の説明をしていきます。

強化学習における**エントロピー（entropy）** は、「行動確率分布のランダムさ（不確実性）」を表す指標です。

### 1. エントロピーの直感的な意味

- **エントロピーが高い**  
  → 行動確率がほぼ均等で、どの行動を選ぶかが**ランダム（不確実）** 。  
  → 例: 上下左右停止の確率がすべて 0.2 ずつ。

- **エントロピーが低い**  
  → ある行動の確率がほぼ 1.0 で、他の行動はほぼ 0。  
  → 例: 「上」の確率が 0.99、他は 0.01 ずつ → **ほぼ決定的（確定的）** 。

### 2. 数式的な定義（Categorical分布の場合）

Actor（ポリシー）が出力する行動確率分布を `p(a)`（例: `[0.2, 0.2, 0.2, 0.2, 0.2]`）とすると、エントロピーは以下のように定義されます。

```math
H(p) = - \sum_{a} p(a) \log p(a)
```

- `p(a)` が均等に近いほど、`H(p)` は**大きくなる**（エントロピーが高い）。
- `p(a)` が1つの行動に集中するほど、`H(p)` は**小さくなる**（エントロピーが低い）。

PyTorchでは、`Categorical` 分布に対して `dist.entropy()` で計算できます。

### 3. 強化学習での役割

__(1) 探索（Exploration）の指標__
- **学習初期**  
  - エントロピーが高い → ランダムに行動を試しており、**探索が活発**。
- **学習が進むと**  
  - エントロピーが徐々に下がる → 良い行動に集中し、**探索が減り、活用（exploitation）が増える**。

__(2) エントロピー正則化（Entropy Regularization）__
PPOなどのアルゴリズムでは、**エントロピーを目的関数に加える**ことがあります。

```math
L = L_{PPO} + \beta H(p)
```

- `β` はエントロピー係数（例: 0.01）。
- **目的**:  
  - エントロピーが高くなるように少し「押し上げる」ことで、**早すぎる収束（局所解へのハマり）を防ぎ、探索を維持**する。

__(3) 学習の安定性の指標__
- **エントロピーが急激に下がる**  
  → ポリシーが急に硬直し、探索がほぼゼロになる可能性。  
  → 学習率が高すぎる・クリップ範囲が狭すぎるなどのサイン。
- **エントロピーが適度に減少しながら安定**  
  → 学習が順調に進み、徐々に良い行動に集中している状態。

### 4. Pursuit＋MAPPOでの具体的な見方

Pursuit環境でMAPPOを学習する場合、エントロピーは以下のように解釈できます。

- **学習初期**  
  - エントロピーが高い → 追跡者がランダムに動き回っている。
- **学習が進むと**  
  - エントロピーが徐々に下がる → 追跡者が「獲物を囲む動き」など、良い行動に集中。
- **エントロピーが低すぎる（ほぼ0）**  
  - 常に同じ動きしかせず、環境変化に対応できない可能性。
  - エントロピー係数 `entropy_coef` を調整するサイン。

## 性能指標の実装

今回はPursuitの環境でエージェントがうまく立ち回ったか、学習がうまく進んだかを確認する指標として
1. 1エピソードあたりの捕獲数
2. Actor/Critic Loss
3. エントロピ
の3点を実装していきます。

### 1. 1エピソードあたりの捕獲数（成功率）

__実装箇所__
- **学習ループ内（1ステップごと）**  
  → `env.step(agent, action)` の直後。

__計算方法__
- Pursuitでは、**逃走者を完全に囲んで捕獲したときに +5 の報酬**が入るので、  
  `reward == 5.0` の回数を数えます。

```python
# 1ステップ進める
reward, terminated, truncated, info = env.step(agent, action)
episode_reward += reward

# 報酬が +5 なら捕獲発生
if reward == 5.0:
    episode_captures += 1
```

__モニタリング__
- エピソード終了後、`episode_captures` を `capture_buffer` に追加し、**移動平均**を出力。

```python
capture_buffer.append(episode_captures)
avg_captures = np.mean(capture_buffer)
print(f"Avg Captures: {avg_captures:.2f}")
```

### 2. Actor/Critic Loss の推移

__実装箇所__
- **MAPPO.update メソッド内**  
  → PPO更新の各エポックで損失を計算し、**平均を返す**。

__計算方法__
- `actor_loss`（PPOクリップ付き損失）と `critic_loss`（MSE）を各エポックで記録し、平均を返す。

```python
# 損失・エントロピーの記録用リスト
actor_losses = []
critic_losses = []
entropies = []

for epoch in range(epochs):
    # Actor更新
    dist = self.actor(obs.view(-1, self.obs_dim))
    # ...（PPO損失計算）
    actor_losses.append(actor_loss.item())

    # Critic更新
    values = self.critic(global_states).squeeze()
    critic_loss = nn.MSELoss()(values, returns.mean(dim=1))
    critic_losses.append(critic_loss.item())

# エポック平均を返す
avg_actor_loss = np.mean(actor_losses)
avg_critic_loss = np.mean(critic_losses)
return avg_actor_loss, avg_critic_loss, avg_entropy
```

### モニタリング
- 学習ループ側で `mappo.update(batch)` の返り値を受け取り、**エピソードごとに出力**。

```python
actor_loss, critic_loss, entropy = mappo.update(batch)
print(f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
```

### 3. エントロピーの推移（探索の適切さ）

__実装箇所__
- **MAPPO.update メソッド内（Actor更新部分）**  
  → `dist.entropy().mean()` でバッチ平均のエントロピーを計算。

__計算方法__
- Actorの出力分布（Categorical）からエントロピーを取得し、各エポックで記録。

```python
dist = self.actor(obs.view(-1, self.obs_dim))
entropy = dist.entropy().mean()  # バッチ平均
entropies.append(entropy.item())
```

__モニタリング__
- `update` メソッドが `entropy` を返すようにし、学習ループ側で出力。

```python
actor_loss, critic_loss, entropy = mappo.update(batch)
print(f"Avg Entropy: {entropy:.4f}")
```

### 4. 学習ループ側での統合（まとめ）

```python
# モニタリング用のバッファ
reward_buffer = deque(maxlen=100)
capture_buffer = deque(maxlen=100)

for episode in range(max_episodes):
    # 1エピソード実行（捕獲数・報酬を計測）
    # ...

    # Advantage計算
    mappo.buffer.compute_advantages(gamma=gamma, gae_lambda=gae_lambda)

    # PPO更新（Actor/Critic Loss・エントロピーを計測）
    actor_losses_episode = []
    critic_losses_episode = []
    entropies_episode = []

    for _ in range(update_epochs):
        batch = mappo.buffer.sample(batch_size)
        actor_loss, critic_loss, entropy = mappo.update(batch)

        actor_losses_episode.append(actor_loss)
        critic_losses_episode.append(critic_loss)
        entropies_episode.append(entropy)

    avg_actor_loss = np.mean(actor_losses_episode)
    avg_critic_loss = np.mean(critic_losses_episode)
    avg_entropy = np.mean(entropies_episode)

    # ログ出力
    print(f"Episode {episode}: "
          f"Avg Reward: {np.mean(reward_buffer):.2f} | "
          f"Avg Captures: {np.mean(capture_buffer):.2f} | "
          f"Actor Loss: {avg_actor_loss:.4f} | "
          f"Critic Loss: {avg_critic_loss:.4f} | "
          f"Avg Entropy: {avg_entropy:.4f}")
```

### 5. 実装のポイントまとめ

- **捕獲数**  
  - `reward == 5.0` の回数を数える（Pursuit仕様）。  
  - エピソードごとに `capture_buffer` で移動平均。

- **Actor/Critic Loss**  
  - `MAPPO.update` 内で各エポックの損失を記録し、**平均を返す**。  
  - 学習ループ側でエピソードごとに出力。

- **エントロピー**  
  - `dist.entropy().mean()` でバッチ平均を計算。  
  - `update` メソッドが返すようにし、学習ループ側で出力。

これにより、  
- **タスクの成功度（捕獲数）**  
- **学習の安定性（Loss）**  
- **探索の適切さ（エントロピー）**  
を同時にモニタリングできるようになります。

## 総括
今回はPursuitに性能指標を実装を行い、学習の進み具合や、うまく学習しているかを確認できるようにしました。

**1. 1エピソードあたりの捕獲数（成功率）**
- **実装箇所**: 学習ループ内（`env.step` 直後）。
- **計算方法**: `reward == 5.0` の回数を数える（Pursuitの捕獲報酬）。
- **モニタリング**: `capture_buffer` で移動平均を出力。

**2. Actor/Critic Loss の推移**
- **実装箇所**: `MAPPO.update` 内。
- **計算方法**: 各エポックのPPO損失（Actor）とMSE損失（Critic）を記録し、平均を返す。
- **モニタリング**: 学習ループ側でエピソードごとに出力。

**3. エントロピーの推移（探索の適切さ）**
- **実装箇所**: `MAPPO.update` 内（Actor更新部分）。
- **計算方法**: `dist.entropy().mean()` でバッチ平均のエントロピーを計算。
- **モニタリング**: `update` の返り値として受け取り、エピソードごとに出力。

**まとめ**
- **捕獲数**: タスクの成功度（どれだけ獲物を捕まえられたか）。
- **Loss**: 学習の安定性（Actor/Criticが正しく更新されているか）。
- **エントロピー**: 探索の適切さ（ランダムすぎないか・硬直しすぎていないか）。


