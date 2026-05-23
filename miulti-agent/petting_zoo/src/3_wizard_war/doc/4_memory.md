
先日のAtari 環境（Wizard of Wor）の実装においてネットワークの次はメモリバッファの実装を行います。

## メモリ実装のコツ

強化学習、特に MAPPO のようなマルチエージェント手法におけるメモリ（ロールアウトバッファ）設計は、学習の「速度・安定性・メモリ消費量」に直結します。
Atari 環境を扱う上で、実戦的なコツを整理しました。

![1779506244085](image/4_memory/1779506244085.png)

### 1. メモリ節約：データ型の最適化

Atari の画像データ（210x160x3）を `float32` で保存すると、すぐにメモリ（RAM/VRAM）が枯渇します。

* **`uint8` で保存する:** 画像はバッファ内では `0-255` の整数（`uint8`）で保持し、**学習（ミニバッチ生成）の直前で `float32` に変換して `255.0` で割る**ようにします。これだけでメモリ使用量は **1/4** になります。
* **共有メモリの活用:** もし並列環境（複数のシミュレーションを同時に回す）を使う場合は、Python の `multiprocessing.Array` などを用いてメモリを共有し、無駄なコピーを避けます。


### 2. MAPPO 特有の Tips：集中 Critic 用の情報

MAPPO では、Critic が学習時に「自分以外の情報」を必要とします。

* **Global State の明示的保存:** Wizard of Wor は画面全体が見えるため個人の観測（Local Obs）と全体（Global State）がほぼ同じですが、将来的に「自分にしか見えない情報」がある環境に拡張する場合、`state` 用の領域を別途確保しておくと汎用性が高まります。
* **Agent ID のインデックス化:** One-hot ベクトルをそのまま保存するのではなく、整数（0 または 1）として保存し、学習時に `F.one_hot` で変換するとメモリ効率が良いです。

### 3. 学習の安定化：正規化（Normalization）

PPO/MAPPO の論文で強く推奨されている手法です。

* **報酬の正規化（Reward Scaling）:** 報酬をそのまま入れるのではなく、これまでの報酬の移動平均や標準偏差でスケールします。Atari はゲームによってスコアの桁が全く違うため、これは必須級の処理です。
* **アドバンテージの標準化:** バッファからミニバッチを取り出す際、そのミニバッチ内のアドバンテージの平均を 0、標準偏差を 1 に正規化します。これにより、Actor の更新が安定します。

### 4. 高度な Tips：Frame Stacking（フレームスタック）

Atari ゲームにおいて、「弾がどの方向に飛んでいるか」「敵がどちらに動いているか」を認識するには、1枚の画像では不十分です。

* **バッファ内での工夫:** 直近 4 フレームを結合して入力する場合、バッファに「4フレーム分を結合したデータ」をそのまま保存すると重複が多く、メモリが 4倍必要になります。
* **「ポインタ」による管理:** 最新の 1 フレームだけを保存し、学習時にインデックスを遡って 4 フレーム分取り出すように設計すると、メモリ効率が劇的に向上します。

### 5. デバッグのための Tips：Mask（マスク）の扱い

* **Termination vs Truncation:** エージェントが死んだ（Termination）のか、時間切れ（Truncation）なのかを区別して保存します。
* **Dead Agent の処理:** 片方のエージェントが先に脱落した場合、そのエージェントの `mask` を `0` にして、その後のデータが勾配計算に悪影響を与えないように設計します。


### メモリ構成の推奨データ構造例

| データ項目 | 型 | 形状 (Shape) | 備考 |
| --- | --- | --- | --- |
| `obs` | `uint8` | `(Steps, Agents, 3, 210, 160)` | 学習直前に `/255.0` |
| `actions` | `int64` | `(Steps, Agents)` |  |
| `rewards` | `float32` | `(Steps, Agents)` | 報酬スケーリング推奨 |
| `values` | `float32` | `(Steps, Agents)` | Critic の予測値 |
| `masks` | `float32` | `(Steps, Agents)` | 生存なら 1.0, 終了なら 0.0 |
| `log_probs` | `float32` | `(Steps, Agents)` | サンプリング時の対数確率 |

## メモリの実装

MAPPOの学習では、エージェント全員の「観測、行動、報酬、ログ確率、価値」などを保存する、共有ロールアウトバッファ（Shared Rollout Buffer）が必要です。

特にMAPPOは「集中Critic」を用いるため、各エージェントの個別データだけでなく、学習時に必要なグローバル情報も一緒に保持できる構造が望ましいです。

### 1. メモリ設計のポイント

MAPPOの学習（PPOアルゴリズム）はオンポリシー（On-Policy）であるため、以下のサイクルで動作します。

1. **収集:** 一定ステップ（例：128〜2048ステップ）分のデータをバッファに溜める。
2. **学習:** バッファ内のデータを使って数エポック更新を行う。
3. **破棄:** 学習が終わったらデータを全て捨て、また収集に戻る。

### 2. MAPPO用共有ロールアウトバッファの実装

```python
import torch
import numpy as np

class MAPPORolloutBuffer:
    def __init__(self, buffer_size, num_agents, obs_shape, action_dim):
        """
        buffer_size: 1回の学習までに溜めるステップ数
        num_agents: エージェント数 (Wizard of Worなら 2)
        obs_shape: 画像のサイズ (3, 210, 160)
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        
        # データの格納場所 (PyTorchテンソルで確保)
        # すべて [ステップ数, エージェント数, 次元] の形に揃える
        self.obs = torch.zeros((buffer_size, num_agents, *obs_shape))
        self.actions = torch.zeros((buffer_size, num_agents))
        self.log_probs = torch.zeros((buffer_size, num_agents))
        self.rewards = torch.zeros((buffer_size, num_agents))
        self.values = torch.zeros((buffer_size, num_agents))
        self.masks = torch.ones((buffer_size, num_agents)) # 終了判定用 (dones)
        
        self.step = 0

    def insert(self, obs, actions, log_probs, values, rewards, masks):
        """
        1ステップ分の全エージェントデータを一括挿入
        obs: {agent_id: tensor} のような辞書、または [num_agents, C, H, W] のテンソル
        """
        # ここでは辞書からテンソルに変換して格納する例
        for i, agent_id in enumerate(['first_0', 'second_0']):
            self.obs[self.step, i] = obs[agent_id]
            self.actions[self.step, i] = actions[agent_id]
            self.log_probs[self.step, i] = log_probs[agent_id]
            self.values[self.step, i] = values[agent_id]
            self.rewards[self.step, i] = rewards[agent_id]
            self.masks[self.step, i] = masks[agent_id]
            
        self.step = (self.step + 1) % self.buffer_size

    def clear(self):
        """学習後にポインタをリセット"""
        self.step = 0

    def get_generator(self, num_mini_batches, advantages, returns):
        """
        学習用のミニバッチを生成するジェネレータ
        """
        batch_size = self.buffer_size * self.num_agents
        mini_batch_size = batch_size // num_mini_batches
        
        # データを平坦化 (flatten) してシャッフル
        # [Step, Agent, ...] -> [Step * Agent, ...]
        flat_obs = self.obs.view(-1, *self.obs.shape[2:])
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        # エージェントIDのOne-hotもフラットに作成
        # [Step, Agent, ID_dim]
        ids = torch.eye(self.num_agents).repeat(self.buffer_size, 1, 1).view(-1, self.num_agents)

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            yield (
                flat_obs[idx],
                ids[idx],
                flat_actions[idx],
                flat_log_probs[idx],
                flat_values[idx],
                flat_advantages[idx],
                flat_returns[idx]
            )

```

### 3. なぜこの設計なのか

* **集中Criticへの対応:** Criticの学習には「その時の状態の価値（Value）」と「実際に得られた収益（Return）」のペアが必要です。このバッファはエージェントごとのValueを並列に持つため、MAPPOの集中評価を効率的に扱えます。
* **フラット化（Flatten）のメリット:** パラメータ共有を行っている場合、プレイヤー1のデータもプレイヤー2のデータも「1つのネットワークを更新するためのサンプル」として同等に扱えます。そのため、ミニバッチ作成時に `Step * Agent` でまとめてシャッフルすることで、学習が安定します。
* **アドバンテージの計算:** PPOでは、バッファがいっぱいになった後に、逆方向に計算して `Advantages`（期待値よりどれだけ良かったか）と `Returns`（割引報酬和）を算出します。このバッファはその計算結果を後付けで受け取れるよう設計しています。

__実装の注意点__

__1. データの「多次元構造」と「フラット化」の同期__

MAPPOのバッファは通常 `[Step, Agent, C, H, W]` という 5 次元の形状を持ちます。これを学習時に `[Step * Agent, C, H, W]` へ変換（フラット化）する際、**他のデータも全く同じ順序で並んでいること**を保証しなければなりません。

* **`reshape` と `view` の使い分け:**
`view` はメモリが連続（contiguous）であることを前提とします。`insert` を繰り返すとメモリ配置が複雑になることがあるため、安全のために `reshape` を使うか、事前に `.contiguous()` を呼び出します。
* **ID（エージェント識別子）の生成:**
今回のように `torch.eye(num_agents).repeat(buffer_size, 1)` を使う手法は非常に安全です。これにより `[A1, A2, A1, A2, ...]` という順序が確定し、`obs.reshape` 後の順序と確実に一致します。

__2. インデックス管理の厳密化__

`insert` メソッド内で「どのデータがどのエージェントのものか」を曖昧にすると、学習が崩壊します。

* **辞書キーの固定:**
`obs['first_0']` のようにキーで受け取る場合、ループ内で `enumerate(['first_0', 'second_0'])` を使ってインデックス（`i=0, 1`）を明示的に割り当て、バッファの `Agent` 次元に固定して格納します。
* **挿入と更新のタイミング:**
`self.step`（書き込み位置）の更新は、**全エージェントのデータを書き込み終わった最後**に行います。ループ内で行うと、エージェント間でステップがズレる原因になります。

__3. 型変換と正規化のベストプラクティス__

画像データ（特に Atari や Wizard of Wor）を扱う場合、メモリ効率と精度のバランスが重要です。

* **`uint8` での保存:**
バッファ内では `float32` (0.0-1.0) ではなく `uint8` (0-255) で保持することで、メモリ使用量を 4 分の 1 に抑えられます。
* **取り出し時の正規化:**
`get_generator` でデータを取り出した直後に `.float() / 255.0` を行います。
* **注意:** テストコードで検証する際は、この「正規化後の値」で判定しているか、「正規化前の値」で判定しているかを意識しないと、今回のような判定エラー（`val=1.000` なのに `threshold=100` で判定など）が発生します。




### 4. 実装のtest

バッファの並び順、ミニバッチのフラットが出来ているか以下のtest用コードで確認します。

```python
import torch
import numpy as np

def test_buffer_integrity():
    # --- 1. 設定 ---
    buffer_size = 10
    num_agents = 2
    obs_shape = (3, 210, 160)
    action_dim = 9 # Wizard of Wor の標準的なアクション数
    
    buffer = MAPPORolloutBuffer(buffer_size, num_agents, obs_shape, action_dim)
    
    print(f"Testing Buffer: Size={buffer_size}, Agents={num_agents}")

    # --- 2. ダミーデータの挿入 ---
    # 連続的な値を入れて、後で取り出した時に順番や対応が正しいか確認する
    for s in range(buffer_size):
        # 観測データ：エージェントごとに明確に違う値を入れる
        obs = {
            'first_0': torch.full(obs_shape, 10.0),  # P1は常に 10 (255で割ると約0.04)
            'second_0': torch.full(obs_shape, 200.0) # P2は常に 200 (255で割ると約0.78)
        }
        actions = {'first_0': 1, 'second_0': 2}
        log_probs = {'first_0': -0.5, 'second_0': -0.6}
        values = {'first_0': 10.0, 'second_0': 11.0}
        rewards = {'first_0': 1.0, 'second_0': 0.0}
        masks = {'first_0': 1.0, 'second_0': 1.0}
        
        buffer.insert(obs, actions, log_probs, values, rewards, masks)

    print("Data insertion completed.")

    # --- 3. ミニバッチ生成のテスト ---
    # GAE計算後を想定したダミーのアドバンテージとリターン
    dummy_advantages = torch.randn(buffer_size, num_agents)
    dummy_returns = torch.randn(buffer_size, num_agents)
    
    num_mini_batches = 2
    generator = buffer.get_generator(num_mini_batches, dummy_advantages, dummy_returns)
    
    batch = next(generator)
    obs_b, ids_b, act_b, lp_b, val_b, adv_b, ret_b = batch

    # --- 4. 判定ロジック ---
    errors = []

    # A. 形状チェック
    expected_batch_size = (buffer_size * num_agents) // num_mini_batches
    if obs_b.shape != (expected_batch_size, *obs_shape):
        errors.append(f"Obs shape mismatch: {obs_b.shape}")
    if ids_b.shape != (expected_batch_size, num_agents):
        errors.append(f"IDs shape mismatch: {ids_b.shape}")

    # B. IDとデータの対応チェック
    # Player 1 (ID [1,0]) なら値は 100未満、Player 2 (ID [0,1]) なら 100以上のはず
    for i in range(expected_batch_size):
        # バッファ内で /255.0 しているので、
        # P1 は 0.0 以上 1.0 未満 (s / 255)
        # P2 は 100/255.0 (~0.39) 以上 になるはず
        sample_val = obs_b[i].mean().item() # 空間平均をとるのが確実
        is_p1 = ids_b[i, 0] == 1.0 # [1, 0] なら P1
        
        # 判定しきい値を 50/255.0 (約0.19) に設定
        threshold = 0.5 # 中間の 0.5 をしきい値にする
        for i in range(expected_batch_size):
            sample_val = obs_b[i].mean().item()
            is_p1 = ids_b[i, 0] == 1.0 
            
            if is_p1 and sample_val > threshold:
                errors.append(f"Sample {i}: ID is P1 (val={sample_val:.3f}), but data > {threshold}")
            if not is_p1 and sample_val < threshold:
                errors.append(f"Sample {i}: ID is P2 (val={sample_val:.3f}), but data < {threshold}")

    # C. 正規化チェック（正規化ロジックをバッファに入れている場合）
    if obs_b.max() > 1.0 and buffer.obs.max() > 1.0:
        # もしバッファ内で255のままで、取り出し時に割っていないなら警告
        print(" Warning: Observations are not normalized to [0, 1]. Ensure '/ 255.0' is applied.")

    # --- 5. 結論 ---
    if not errors:
        print("\n" + "="*30)
        print(" ALL CHECKS PASSED")
        print(f" Batch Size: {expected_batch_size}")
        print(f" ID-Data Correlation: Confirmed")
        print("="*30)
    else:
        print("\n" + "!"*30)
        print("INTEGRITY ERRORS FOUND ")
        for err in errors:
            print(f" - {err}")
        print("!"*30)

# 実行
test_buffer_integrity()
```

上記を実行して以下のように表示されれば確認クリアです。

```
Testing Buffer: Size=10, Agents=2
Data insertion completed.

==============================
 ALL CHECKS PASSED
 Batch Size: 10
 ID-Data Correlation: Confirmed
==============================
```

## 総括

Atari（Wizard of Wor）＋ MAPPO のロールアウトバッファ実装のエッセンスを、さらにコンパクトにまとめると次の通りとします。

- **メモリ節約**  
  - 画像は `uint8` で保存し、ミニバッチ生成直前に `.float() / 255.0` で正規化する。  
  - 並列環境では共有メモリでコピーを避ける。

- **MAPPO 用設計**  
  - 集中 Critic 用に `state` 領域を別途確保。  
  - Agent ID は整数で保存し、学習時に `F.one_hot` で変換する。

- **安定化**  
  - 報酬を移動平均・標準偏差でスケーリング。  
  - ミニバッチ内のアドバンテージを平均 0・標準偏差 1 に正規化。

- **Frame Stacking**  
  - 最新 1 フレームだけ保存し、学習時にインデックスを遡って 4 フレーム分を組み立てる。

- **マスクと終了**  
  - Termination / Truncation を区別。  
  - Dead Agent の `mask` を 0 にして勾配計算から除外。

- **バッファ構造と順序管理**  
  - `[Step, Agent, ...]` → `[Step * Agent, ...]` のフラット化で、`obs`, `actions`, `log_probs`, `values` の順序を完全に一致させる。  
  - `insert` 内でキーとインデックスを固定し、Agent 次元への格納順を厳密に管理。  
  - テストコードで「ID と観測値の対応」「バッチサイズ」「正規化のタイミング」を検証する。



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
