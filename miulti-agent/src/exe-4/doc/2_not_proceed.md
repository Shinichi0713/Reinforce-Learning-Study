# MAPPO

## 全然学習が進まない

学習が進まない原因は、主に**「報酬の希薄さ（Sparse Rewards）」**と、**「PPOとしての時間的学習の欠如」**にあります。

現在の実装では、荷物を偶然拾って届けるまで（数十〜百ステップ）報酬がほぼ発生しません。強化学習エージェントは「何が正解か」の手がかりを掴めず、ランダムに震えるだけの動きになります。

以下の**3つの修正**を適用してください。

---

### 1. 報酬設計の改善（Reward Shaping）

これが最も重要です。エージェントが「荷物に近づく」「目的地に近づく」こと自体に小さな報酬を与えます。

`DroneDeliveryEnv`の`step`メソッド内の報酬計算を以下のように書き換えます。

```python
# --- stepメソッド内の報酬計算部分を修正 ---
for i in range(self.num_agents):
    pos = self.agent_pos[i]
    carry = self.agent_has[i]
    
    if carry == -1:
        # 荷物を持っていない時：最も近い「未回収」の荷物への距離を報酬にする
        undelivered_pkgs = [p for p in self.packages if not p[2]]
        if undelivered_pkgs:
            dists = [np.abs(pos[0]-p[0][0]) + np.abs(pos[1]-p[0][1]) for p in undelivered_pkgs]
            # 距離が近いほど報酬（最大0.1程度になるよう調整）
            rewards[i] += 0.01 * (10 - min(dists)) 
    else:
        # 荷物を持っている時：その荷物の目的地への距離を報酬にする
        drop_pos = self.packages[carry][1]
        dist_to_drop = np.abs(pos[0]-drop_pos[0]) + np.abs(pos[1]-drop_pos[1])
        rewards[i] += 0.01 * (10 - dist_to_drop)

```

---

### 2. PPOとしての割引報酬の計算

現在のコードは「その瞬間の報酬」をターゲットにしていますが、PPOでは**「将来にわたる報酬の合計（割引報酬和）」**を学習する必要があります。`train`メソッドの冒頭に以下を追加してください。

```python
    def train(self, memory):
        if len(memory.states) == 0: return

        # --- 割引報酬和 (Returns) の計算を追加 ---
        raw_rewards = torch.stack(memory.returns) # (T, Agents)
        T = raw_rewards.size(0)
        discounted_returns = torch.zeros_like(raw_rewards)
        
        # 簡易的な累積報酬の計算
        running_add = torch.zeros(self.num_agents)
        for t in reversed(range(T)):
            running_add = raw_rewards[t] + self.gamma * running_add
            discounted_returns[t] = running_add
            
        # テンソル展開
        states = torch.stack(memory.states)
        obs = torch.stack(memory.obs)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs).detach() # 重要
        returns = discounted_returns # 修正した累積報酬を使用

```

---

### 3. データ収集時の「計算グラフ」の切り離し

先程のエラー対策でも触れましたが、`memory`に保存するデータは完全に勾配から切り離す必要があります。学習ループ（`for t in range(env.max_steps):`）の中を以下のように徹底してください。

```python
        # 推論は勾配計算なしで
        with torch.no_grad():
            obs_t = trainer._obs_to_tensor(obs_list)
            state_t = obs_t.view(-1)
            
            actions, log_probs = [], []
            for i in range(2):
                dist = trainer.actors[i](obs_t[i])
                a = dist.sample()
                actions.append(a.item())
                log_probs.append(dist.log_prob(a))
        
        next_obs_list, rewards, done, _ = env.step(actions)
        
        # 保存時に必ず .detach() をつける
        memory.obs.append(obs_t.detach())
        memory.states.append(state_t.detach())
        memory.actions.append(torch.tensor(actions).detach())
        memory.log_probs.append(torch.stack(log_probs).detach())
        memory.returns.append(torch.FloatTensor(rewards).detach())

```

### なぜこれで進むようになるのか？

* **Reward Shaping**: エージェントがランダムに動いた際、「荷物に1歩近づいた」だけで報酬が貰えるため、Actorネットワークの勾配が正しい方向（荷物の方向）に向き始めます。
* **割引報酬和**: 荷物を「拾う」という行動が、後の「届けて得られる大報酬」に繋がっていることを Critic が理解できるようになります。

**次の確認ステップ:**
まず、エピソードごとの報酬合計（Reward）が、**初期のマイナス（衝突）から、徐々にプラス方向に増え始めるか**を確認してください。もし数千エピソード回しても増えない場合は、エントロピー係数（`0.01`の部分）を `0.05` に上げて探索を強めるのが有効です。

学習が少し進んだ段階のGIFを保存してみませんか？ドローンが荷物を「意識して追いかけている」様子が見えるはずです。


