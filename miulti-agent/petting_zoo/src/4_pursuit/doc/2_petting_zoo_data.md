昨日のpettingの[Pursuit](https://yoshishinnze.hatenablog.com/entry/2026/06/06/043000)について学習するプログラムの実装を進めていきます。

## 実装の進め方

PursuitをMAPPOで「初めから正しく、わかりやすく」実装するには、**役割ごとにステップを分けて進める**のがポイントです。

以下、実装の手順をステップバイステップで整理します。

### ステップ0：準備（環境とライブラリ）

1. PettingZooのPursuit環境が動くことを確認
2. `numpy`, `torch`（または `jax`）, `gymnasium`, `pettingzoo[sisl]` などをインストール
3. プロジェクト用のディレクトリ・ファイル構成を決める（後述）

※ここはGoogle Colabを使うと`ettingzoo`関係のパッケージのみで可です。

```
# エミュレータとGymnasiumのインストール
!pip install gymnasium[atari]
!pip install gymnasium[accept-rom-license]
!pip install pyvirtualdisplay > /dev/null 2>&1
!pip install AutoROM[accept-rom-license]
!AutoROM --accept-license
# 描画用のシステムパッケージ
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install pettingzoo[atari,accept-rom-license]
!pip install "pettingzoo[sisl]"
!pip install supersuit
```

### ステップ1：環境ラッパの実装

**目的**: Pursuit環境を、MAPPOが扱いやすい形に変換する。

1. **環境インスタンスの生成**
   - `env = pursuit_v4.env(render_mode="rgb_array", max_cycles=500, ...)`
   - `shared_reward=True` に設定（協調型なので全員で報酬共有）

2. **観測の前処理**
   - Pursuitの観測は `(7, 7, 3)` のローカルグリッド
   - MAPPOでは
     - CNNで処理する（画像として扱う）
     - もしくはフラット化してMLPで処理する
   - ここでは**フラット化**を例にします：
     ```python
     obs_flat = obs.reshape(-1)  # (7*7*3,) のベクトル
     ```

3. **グローバル状態の構成**
   - Criticには「全エージェントの観測をまとめたもの」を入力したい
   - 簡易的には「全エージェントの観測を結合したベクトル」をグローバル状態とみなす：
     ```python
     global_state = np.concatenate([obs_flat for each_agent])
     ```

4. **エージェントIDの扱い**
   - 各エージェントに一意のID（one-hotなど）を観測に追加すると、1つのポリシーで複数エージェントを区別できます。

5. **ラッパクラスの作成**
   - `reset()` → 各エージェントの観測とグローバル状態を返す
   - `step(action_dict)` → 各エージェントの次観測・報酬・終了フラグ・グローバル状態を返す

### ステップ2：ネットワーク設計（Actor / Critic）

**目的**: MAPPO用のニューラルネットを定義する。

1. **Actorネットワーク（ポリシー）**
   - 入力: エージェント観測（＋ID）
   - 出力: 行動確率分布（softmax）
   - 例（PyTorch）:
     ```python
     class Actor(nn.Module):
         def __init__(self, obs_dim, act_dim, hidden_size=64):
             super().__init__()
             self.fc = nn.Sequential(
                 nn.Linear(obs_dim, hidden_size),
                 nn.ReLU(),
                 nn.Linear(hidden_size, hidden_size),
                 nn.ReLU(),
                 nn.Linear(hidden_size, act_dim)
             )
         def forward(self, obs):
             logits = self.fc(obs)
             return Categorical(logits=logits)
     ```

2. **Criticネットワーク（価値関数）**
   - 入力: グローバル状態
   - 出力: スカラーの状態価値 V(s)
   - 例:
     ```python
     class Critic(nn.Module):
         def __init__(self, state_dim, hidden_size=64):
             super().__init__()
             self.fc = nn.Sequential(
                 nn.Linear(state_dim, hidden_size),
                 nn.ReLU(),
                 nn.Linear(hidden_size, hidden_size),
                 nn.ReLU(),
                 nn.Linear(hidden_size, 1)
             )
         def forward(self, state):
             return self.fc(state)
     ```

### ステップ3：バッファ（経験再生）の設計

**目的**: 1エピソード分のデータを保存し、PPO更新に使える形に整える。

1. **保存するデータ**
   - 観測 `obs`
   - 行動 `actions`
   - 報酬 `rewards`
   - 終了フラグ `dones`（terminated / truncated）
   - ログ確率 `log_probs`
   - 状態価値 `values`（Criticの出力）
   - 必要に応じて `advantages`, `returns`

2. **バッファクラスの例**
   - `add(obs, action, reward, done, log_prob, value)`
   - `compute_advantages_returns()`: GAEでadvantageとreturnを計算
   - `get_batch()`: 学習用にテンソルとして返す

### ステップ4：学習ループ（MAPPOのコア）

**目的**: 上記の部品を組み合わせてMAPPOの学習を行う。

1. **大まかな流れ**
   ```python
   for episode in range(num_episodes):
       # 1. データ収集（ロールアウト）
       obs = env.reset()
       while not done:
           action, log_prob, value = policy(obs)  # Actor & Critic
           next_obs, reward, done, info = env.step(action)
           buffer.add(obs, action, reward, done, log_prob, value)
           obs = next_obs

       # 2. AdvantageとReturnの計算
       buffer.compute_advantages_returns()

       # 3. PPO更新（複数エポック）
       for epoch in range(ppo_epochs):
           batch = buffer.get_batch()
           loss = ppo_update(batch)  # ActorとCriticの損失を計算して更新

       buffer.clear()
   ```

2. **PPO更新の要点**
   - Actor損失:  
     - 比率 `r_t = π_new(a|s) / π_old(a|s)` を使い、`clip(r_t * A, 1-ε, 1+ε)` でクリップ
   - Critic損失:  
     - `(V(s) - Return)^2` のMSE
   - エントロピー項を加えると探索が促進されます

### ステップ5：評価と可視化

**目的**: 学習したポリシーがPursuitをどの程度解けているかを確認する。

1. **報酬の推移をプロット**
   - 各エピソードの平均報酬を記録し、`matplotlib` でグラフ化

2. **動画保存**
   - 一定間隔で学習済みポリシーでPursuitを実行し、GIFやMP4として保存
   - ランダムエージェントとの比較も行うとわかりやすいです

3. **観測の可視化（任意）**
   - 特定のエージェントがどのような観測を見ているかを画像で確認

### まとめ
ということをまとめると以下のように進めることとします。

1. **環境ラッパ**でPursuitをMAPPO向けに整形
2. **Actor/Criticネットワーク**を設計
3. **バッファ**で経験を保存・計算
4. **学習ループ**でPPO更新を繰り返す
5. **評価・可視化**で性能を確認

まずは1. 環境ラッパの作成を行います。

## 環境ラッパの実装

### 環境ラッパの役割
環境ラッパの役割は、**Pursuit環境をMAPPOが扱いやすい形に変換すること**です。具体的には：

- **観測の整形**  
  Pursuitの観測（ローカルグリッド）を、MAPPOのネットワークに入力しやすい形（フラットベクトルやCNN入力など）に変換する。

- **グローバル状態の構成**  
  Critic（価値関数）に入力するための「全エージェントの情報をまとめた状態」を作る。

- **報酬の扱い**  
  協調型タスクとして、全エージェントで報酬を共有する設定（`shared_reward=True`）を適用する。

- **APIの統一**  
  PettingZooのマルチエージェントAPIを、MAPPOの学習ループから呼び出しやすい単一のインターフェースにまとめる。

つまり、**「Pursuitの生データを、MAPPOがそのまま使える形に整える」** のが環境ラッパの役割です。

### 実装のキーポイント

Pursuit環境をMAPPOで学習するための環境ラッパ実装のキーポイントを、**設計方針・実装の要点・よくあるエラー対処**に分けて整理します。

__1. 設計方針（ラッパの役割）__

__(1) ラッパの役割を明確に絞る__
- **観測整形**：Pursuitの `(7,7,3)` 観測をフラット化（`147次元`）し、MAPPOのネットワークに入力しやすい形にする。
- **グローバル状態の構成**：全エージェントの観測を結合して `1176次元` のグローバル状態を作る（MAPPOのCritic用）。
- **APIの統一**：MAPPO側が扱いやすいインターフェース（`get_obs()`, `get_global_state()`, `step(agent, action)`）を提供する。

__(2) `env.agent_iter()` のループはMAPPO側で回す__
- PettingZooのAEC環境は、**`env.agent_iter()` でエージェントを順番に返す**設計です。
- ラッパ内でイテレータを管理しようとすると、状態管理が複雑になり、`KeyError` や `TypeError` の原因になります。
- 安全な設計：
  - ラッパは「**観測整形とグローバル状態の構成**」に専念
  - `env.agent_iter()` のループは **MAPPOの学習ループ側で回す**

__2. 実装のキーポイント__

__(1) 観測整形：`get_obs(agent)`__
- `env.last(agent)` から `obs` を取得し、`reshape(-1)` でフラット化。
- `obs` が `None` の場合は `None` を返す（deadエージェント対応）。
- 例：
  ```python
  def get_obs(self, agent):
      if agent not in self.env.agents:
          return None
      obs, _, _, _, _ = self.env.last(agent)
      if obs is None:
          return None
      obs_flat = obs.reshape(-1).astype(np.float32)
      return obs_flat
  ```

__(2) グローバル状態：`get_global_state()`__
- `self.possible_agents` から全エージェントをループし、`agent in self.env.agents` で存在確認。
- 存在するエージェントの観測を `np.concatenate()` で結合。
- 例：
  ```python
  def get_global_state(self):
      obs_list = []
      for agent in self.possible_agents:
          if agent in self.env.agents:
              obs = self.get_obs(agent)
              if obs is not None:
                  obs_list.append(obs)
      if not obs_list:
          return None
      global_state = np.concatenate(obs_list).astype(np.float32)
      return global_state
  ```

__(3) `step(agent, action)`：deadエージェントと存在確認__
- **deadエージェントには `action=None` しか許されない**（PettingZooの仕様）。
- `env.last(agent)` を呼ぶ前に `agent in self.env.agents` で存在確認。
- `env.step()` の後も再度存在確認（`step` で削除される可能性あり）。
- 例：
  ```python
  def step(self, agent, action):
      # 存在確認
      if agent not in self.env.agents:
          return 0.0, True, True, {}
      
      # dead チェック
      _, _, terminated, truncated, _ = self.env.last(agent)
      if terminated or truncated:
          step_action = None
      else:
          step_action = action
      
      self.env.step(step_action)
      
      # 再度存在確認
      if agent not in self.env.agents:
          return 0.0, True, True, {}
      
      _, reward, terminated, truncated, info = self.env.last(agent)
      return reward, terminated, truncated, info
  ```

__(4) `reset()` はシンプルに__
- `env.reset()` だけ呼び、`agent_iter` のイテレータはMAPPO側で作る。
- 例：
  ```python
  def reset(self):
      self.env.reset()
  ```

__3. よくあるエラーと対処__

__(1) `TypeError: 'NoneType' object is not subscriptable`__
- `env.reset()` の戻り値を `next_agent = env.reset()` のように扱おうとした場合に発生。
- 修正：`env.reset()` は戻り値を持たないので、`agent_iter` はMAPPO側で作る。

__(2) `AssertionError: action is not in action space`__
- `env.step(action)` を呼ぶタイミングがずれ、`env.agent_iter()` の現在エージェントと一致していない。
- 修正：`env.agent_iter()` のループ内で、`env.step(action)` を呼ぶ。

__(3) `KeyError: 'pursuer_0'`__
- deadエージェントが `env.agents` から削除された後、`env.last(agent)` を呼んだ。
- 修正：`env.last(agent)` を呼ぶ前に `agent in env.agents` で存在確認。

__(4) `TypeError: 'AECOrderEnforcingIterable' object is not an iterator`__
- `env.agent_iter()` を `next()` で回そうとしたが、イテレータ化していない。
- 修正：`agent_iter = iter(env.agent_iter())` としてから `next(agent_iter)` を呼ぶ。

__4. MAPPO側での使い方（安全なループ）__

```python
env = PursuitWrapper(render_mode="rgb_array", max_cycles=500)

# リセット
env.reset()

# MAPPO側で agent_iter を回す
for agent in env.env.agent_iter():
    # 観測を取得（存在確認あり）
    obs = env.get_obs(agent)
    global_state = env.get_global_state()
    
    # dead かどうかを確認（存在確認あり）
    if agent not in env.env.agents:
        action = None
    else:
        _, _, terminated, truncated, _ = env.env.last(agent)
        is_dead = terminated or truncated
        if is_dead:
            action = None
        else:
            # ここでMAPPOのポリシーに基づき行動を選択
            action = env.action_space.sample()  # 例: ランダム
    
    # 1ステップ進める（ラッパ内で存在確認あり）
    reward, terminated, truncated, info = env.step(agent, action)
    
    # バッファに保存など...
    # (obs, action, reward, terminated, truncated, ...)

env.close()
```

### 実装の確認

実際に作ったラッパーは以下のレポジトリに保存しています。

https://github.com/Shinichi0713/Reinforce-Learning-Study/tree/main/miulti-agent/petting_zoo/src/4_pursuit

そして実装コードは実際にループで稼働するか確認してみます。

```python
env = PursuitWrapper(render_mode="rgb_array", max_cycles=500)

# リセット
env.reset()

# MAPPO側で agent_iter を回す
for agent in env.env.agent_iter():
    # 観測を取得（存在確認あり）
    obs = env.get_obs(agent)
    global_state = env.get_global_state()
    
    # dead かどうかを確認（存在確認あり）
    if agent not in env.env.agents:
        action = None
    else:
        _, _, terminated, truncated, _ = env.env.last(agent)
        is_dead = terminated or truncated
        if is_dead:
            action = None
        else:
            # ここでMAPPOのポリシーに基づき行動を選択
            action = env.action_space.sample()  # 例: ランダム
    
    # 1ステップ進める（ラッパ内で存在確認あり）
    reward, terminated, truncated, info = env.step(agent, action)
    
    # バッファに保存など...
    # (obs, action, reward, terminated, truncated, ...)

env.close()
```

何もエラーが出なければ、実装は正常に出来たことになります。

## 総括

**PursuitをMAPPOで実装する流れ**

1. **環境ラッパ**
   - Pursuitの観測 `(7,7,3)` をフラット化してActorに入力。
   - 全エージェントの観測を結合してグローバル状態を作り、Criticに入力。
   - `env.agent_iter()` のループはMAPPO側で回し、ラッパは整形とAPI統一に専念。

2. **Actor / Critic**
   - Actor: 観測 → 行動確率分布。
   - Critic: グローバル状態 → 状態価値 V(s)。

3. **バッファ**
   - `obs`, `actions`, `rewards`, `dones`, `log_probs`, `values` を保存。
   - GAEで `advantages` と `returns` を計算し、PPO更新用バッチを返す。

4. **学習ループ**
   - ロールアウト → Advantage/Return計算 → PPO更新（クリップ付き比率＋MSE）を繰り返す。

5. **評価・可視化**
   - 報酬推移のプロット、学習済みポリシーでの動画保存、ランダムエージェントとの比較。

**環境ラッパの要点**
- `get_obs(agent)`: `env.last(agent)` から観測を取得し `reshape(-1)`。
- `get_global_state()`: 全エージェントの観測を `np.concatenate`。
- `step(agent, action)`: `agent in env.agents` で存在確認し、deadなら `action=None`。
- `reset()`: `env.reset()` だけ呼び、`agent_iter` はMAPPO側で管理。
- よくあるエラーは「存在確認不足」と「`agent_iter` の扱いミス」がほとんどなので、そこを丁寧に実装する。

**まとめ**
- 環境ラッパでPursuitをMAPPO向けに整形 → Actor/Critic・バッファ・学習ループを順に実装、というステップで進めると、初めから正しく・わかりやすくMAPPOを組めます。


最期に強化学習の理論をしっかり学びたいという読者の方へおすすめの本です。


<div class="shop-card">
<div class="shop-card-image"><img src="https://m.media-amazon.com/images/I/81lem2peqFL._SL1500_.jpg" alt="商品画像" /></div>
<div class="shop-card-content">
<div class="shop-card-title">強化学習 (機械学習プロフェッショナルシリーズ)</div>
<div class="shop-card-description">同シリーズで緑本のPythonによる強化学習の本を何度も何度も読んだのですが、どうしても読み進めません。試しにと思って3年前に買ったこの本を読み返してみるとすっと読めました。 これからのコーディングは生成AIが書いてくれるのだから、難しい理論本で勉強してコーディングはお任せ（直すべき所は直す）というのが正解なのかもしれない。。。</div>
<div class="shop-card-link"><a href="https://www.amazon.co.jp/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E6%A3%AE%E6%9D%91%E5%93%B2%E9%83%8E-ebook/dp/B07XJXMQGD?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&amp;crid=2Q7JANDTXMDRQ&amp;dib=eyJ2IjoiMSJ9.YZxuAtwvMTmksETM7b4V5tEFcZKwS3FH_fG2YEbWKvrGjHj071QN20LucGBJIEps.GCkT5rik7rfwPmJpLUkBFsUfiUvfOc-QO8WH5HT0oSA&amp;dib_tag=se&amp;keywords=MARL+%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92&amp;qid=1777879215&amp;sprefix=marl+%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%2Caps%2C165&amp;sr=8-1&amp;linkCode=ll2&amp;tag=yoshishinnze-22&amp;linkId=a3ac27efe00549a8b95a7d948fa658b0&amp;ref_=as_li_ss_tl" target="_blank" rel="noopener">Amazonで詳細を見る</a></div>
</div>
</div>
<p>[blog:g:4207112889963697807:banner]</p>
<p>[blog:g:10328749687175353006:banner]</p>
<p>[blog:g:11696248318754550880:banner]</p>
<p>[blog:g:11696248318754550877:banner]</p>

