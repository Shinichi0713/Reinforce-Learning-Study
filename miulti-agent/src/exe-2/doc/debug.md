デバッグお疲れ様です！`ipdb`での確認結果から、エラーの決定的な原因が特定できました。

### 🔍 エラーの原因：リプレイメモリへの保存順序の不一致

`ipdb> p obs` の結果が `{0: -0.1, 1: -0.1}` となっていることから、**本来「観測（座標など）」が入るべき場所に「報酬（数値）」が渡されています。**

原因は、`run_qmix_training` 内での `push` の引数の順番と、`learn` メソッド内でそれを取り出す際のインデックスの順番が**逆転している**ことです。

1. 保存時 (run_qmix_training):
   agent.memory.push(obs, actions, next_obs, rewards, terminated_flag)
   * Index 2: `next_obs`
   * Index 3: **`rewards`** (← ここに辞書 `{0: -0.1, ...}` が入る)
2. 学習時 (IntegratedQMixAgent.learn):
   next_state = torch.cat([self._obs_to_tensor(t[3], is_state=True) for t in batch], dim=0)
   * **`t[3]`** を観測データとして `_obs_to_tensor` に渡しています。
   * しかし、`t[3]` には**報酬（floatの辞書）**が入っているため、`obs[i][0]`（つまり `-0.1[0]`）を実行しようとして `TypeError` が発生しました。

---

### 🛠️ 修正方法

以下の2箇所を修正してください。

#### 1. `run_qmix_training` の `push` 順序を修正

一般的によく使われる `(状態, 行動, 報酬, 次の状態, 終了フラグ)` の順番に合わせます。

**Python**

```
# --- run_qmix_training 関数内 ---

# 2. 環境ステップ
next_obs, rewards, done, info = env.step(actions)
terminated_flag = all(done.values())

# 3. リプレイバッファに保存 (順序を rewards -> next_obs に修正)
agent.memory.push(obs, actions, rewards, next_obs, terminated_flag) # ここを修正！
```

#### 2. `_obs_to_tensor` の `else` ブロックも修正

前回 `is_state=True` の方は修正しましたが、`else`（個別エージェントの観測）の方も同様の書き方に直しておかないと、後で同じエラーが出ます。

**Python**

```
# --- IntegratedQMixAgent._obs_to_tensor メソッド内 ---

    def _obs_to_tensor(self, obs: Dict[int, Tuple], is_state: bool = False):
        if is_state:
            state_vec = []
            for i in range(self.n_agents):
                pos_tuple = obs[i][0]
                state_vec.extend([pos_tuple[0] / (GRID_SIZE - 1), pos_tuple[1] / (GRID_SIZE - 1)])
                state_vec.append(1.0 if obs[i][1] else 0.0)
          
            # 残り注文の処理
            remaining_orders_set = set(obs[0][2])
            for order_idx in range(NUM_ORDERS):
                state_vec.append(1.0 if order_idx in remaining_orders_set else 0.0)
            return torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
      
        else:
            tensors = {}
            for i in range(self.n_agents):
                # --- ここも修正：リスト内包表記ではなく明示的なインデックス指定に ---
                pos_tuple = obs[i][0]
                obs_i = [pos_tuple[0] / (GRID_SIZE - 1), pos_tuple[1] / (GRID_SIZE - 1)]
              
                obs_i.append(1.0 if obs[i][1] else 0.0)
                agent_id_vec = [0.0] * self.n_agents
                agent_id_vec[i] = 1.0
                obs_i.extend(agent_id_vec)
                tensors[i] = torch.FloatTensor(obs_i).to(self.device).unsqueeze(0)
            return tensors
```

#### 3. 必要なインポートの追加

コード内で `F.elu` や `F.mse_loss` を使用している場合、以下のインポートが必要です。

**Python**

```
import torch.nn.functional as F
```

---

### 💡 補足：現在の `WarehouseEnv.step` について

現在のコードでは `step` メソッドがダミー報酬 (`-0.1`) を返すようになっています。

**Python**

```
def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
    next_obs = self._get_obs()
    rewards = {i: -0.1 for i in range(self.num_agents)} # 全員常にマイナス
    # ...
```

このままだとエージェントは「何もしない」のが正解だと学習してしまう可能性があるため、動作確認ができたら以前作成した「荷物を運んだらプラス報酬」のロジックを `step` 内に戻してください。

まずは上記の **`push` の引数順序の入れ替え** を試してみてください！これでエラーは解消するはずです。

次はこの環境で報酬をより効率的に得るための「報酬設計（シェーピング）」の具体的な実装をお手伝いしましょうか？
