連日Atariのボクシングを強化学習エージェントで勝利するためのアルゴリズム構築を行っています。

__ここまでの関連記事__  
1. petting-zooの導入(MARL環境の導入)
https://yoshishinnze.hatenablog.com/entry/2026/05/09/043000

2. 解くためのアルゴリズム選定

3. データパイプライン構築

## 解法のシナリオ
以下の流れに沿って今回のボクシングゲームの解決を行っていきます。

1. データパイプライン構築: ゲームの画像から「集中クリティック用の入力」を作成し、バッファに正しく保存・取り出しができる仕組みを作ります。
2. ネットワーク設計: 210x160の画像を処理するための畳み込み層を設計します。
3. 学習用エージェント設計: 学習を実際に進めるエージェントを構築します。
4. 強化学習してエージェントにデモプレイします。

今回は3. 学習用エージェント設計を実施していきます。

## GAE
実装の前に必要な知識となる **GAE（一般化アドバンテージ推定）** について説明します。
GAEを一言で言うと、「報酬の受け取り方のバランスを調整して、学習を安定させるテクニック」のことです。
強化学習（特にPPO）では、「この行動はどれくらい良かったのか？」というアドバンテージ（期待以上の良さ）を計算する必要がありますが、その計算には大きなジレンマがあります。

### 1. GAEが解決する「ジレンマ」

アドバンテージを計算する方法は、大きく分けて2つあります。

* **方法A：実際の報酬を最後まで足す (Monte Carlo)**
* **メリット**: 正確（実際に得られた報酬なので）。
* **デメリット**: **バラツキ（分散）が激しい**。たまたま敵が変な動きをしただけで「この行動は最高だ！」と勘違いしやすい。


* **方法B：1ステップ後の予測値を使う (TD誤差)**
* **メリット**: 安定している。
* **デメリット**: **不正確（バイアスがある）**。自分の「予測（Value）」が間違っていると、学習全体が間違った方向に進む。

**GAEは、このAとBを「いいとこ取り」して混ぜ合わせる手法です。**


### 2. 直感的なイメージ：ボクシングでの例

あなたがパンチを打った瞬間を想像してください。

* **1ステップ後**: まだ当たっていませんが、相手との距離が詰まったので「価値（Value）」は上がります。
* **10ステップ後**: 実際にパンチが当たり、報酬 $+1$ を得ました。

GAEは、**「今すぐの結果（TD誤差）」** だけでなく、**「少し先の未来の結果」** も指数関数的に重み付けして合算します。

* $\lambda$（ラムダ）というパラメータを使い、どれくらい未来まで考慮するかを調整します。
* $\lambda = 0$ なら、今の予測（方法B）を信じる。
* $\lambda = 1$ なら、最後まで全部足す（方法A）。
* **通常 $\lambda = 0.95$ くらい**に設定し、ほどよく未来を見つつ、今の安定感も保ちます。

### 3. GAEの計算式（数式）

理論的には、各ステップ $t$ での「TD誤差（予測のズレ）」を $\delta_t$ とすると、GAE（$\hat{A}_t$）は以下のように計算されます。

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots$$

> ※ $\gamma$（ガンマ）は割引率、$\lambda$（ラムダ）がGAEの調整係数です。


### 4. MAPPOにおけるGAEの重要性

マルチエージェント（MAPPO）では、自分だけでなく「相手の動き」という不確定要素が加わるため、報酬のバラツキがさらに激しくなります。

GAEを使うことで：

* **ノイズに強くなる**: 相手の突発的な動きによる報酬の変化を、上手くマイルドに処理できます。
* **効率的な学習**: 集中クリティックが予測した「神の視点での価値 $V(s)$」をベースにGAEを計算するため、非常に質の高いアドバンテージが得られます。


### 次の実装：バッファ内でのGAE計算

次は、この理屈をコードに落とし込みます。具体的には、**バッファを後ろ（最新のステップ）から順に遡りながら、この $\hat{A}_t$ を計算していくロジック**を実装します。
ロジック自体は前回のリプレーバッファに実装することが自然です。

## 学習用エージェント

ボクシング環境でのMAPPO攻略に向けて、これまでに作成した「環境」「ネットワーク」「バッファ」を統括し、学習を回すためのコアエンジンとなる `MAPPOAtariTrainer` クラスを実装します。

このクラスは、データの収集からGAEの計算、そして勾配更新（学習）までを一手に引き受けます。

### MAPPOAtariTrainer クラスの実装

```python
import torch
import torch.optim as optim

class MAPPOAtariTrainer:
    def __init__(self, env, agent, buffer_size=2048, batch_size=64, lr=3e-4, gamma=0.99, gae_lambda=0.95, ppo_epochs=10):
        self.env = env
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)
        
        # ハイパーパラメータ
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # バッファの初期化
        # obs_shape=(4, 84, 84), joint_shape=(8, 84, 84)
        self.buffer = MAPPORolloutBuffer(buffer_size, (4, 84, 84), (8, 84, 84), self.device)
        
        # オプティマイザ (ActorとCriticをまとめて更新)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

    def collect_rollouts(self):
        """環境を動かしてデータを収集する"""
        self.buffer.clear()
        obs_dict, _ = self.env.reset()
        
        for _ in range(self.buffer.buffer_size):
            # 1. 前処理
            o1, o2, joint_s = preprocess_joint_obs(obs_dict, self.device)
            
            # 2. 行動決定と価値予測
            with torch.no_grad():
                a1, logp1, _ = self.agent.get_action(o1.unsqueeze(0))
                a2, logp2, _ = self.agent.get_action(o2.unsqueeze(0))
                v1, v2 = self.agent.get_value(joint_s.unsqueeze(0))
            
            # 3. 環境の実行
            actions = {'first_0': a1.item(), 'second_0': a2.item()}
            next_obs_dict, rewards, terms, truncs, infos = self.env.step(actions)
            
            # 4. バッファへ保存
            dones = [terms['first_0'] or truncs['first_0'], terms['second_0'] or truncs['second_0']]
            self.buffer.insert(
                o1, o2, joint_s, 
                [a1.item(), a2.item()], 
                [logp1.item(), logp2.item()],
                [rewards['first_0'], rewards['second_0']],
                [v1.item(), v2.item()],
                dones
            )
            
            obs_dict = next_obs_dict
            if any(dones):
                obs_dict, _ = self.env.reset()

        # 5. GAEの計算準備 (最後の状態の価値)
        _, _, last_joint_s = preprocess_joint_obs(obs_dict, self.device)
        with torch.no_grad():
            last_v1, last_v2 = self.agent.get_value(last_joint_s.unsqueeze(0))
        
        self.buffer.compute_returns_and_advantages(
            torch.tensor([last_v1.item(), last_v2.item()], device=self.device),
            self.gamma, self.gae_lambda
        )

    def train_step(self, clip_param=0.2, ent_coef=0.01, vf_coef=0.5):
        """バッファのデータを使ってネットワークを更新する"""
        total_loss = 0
        
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # 1Pと2Pのデータをまとめて処理するために変形
                # batch['obs']: (batch, 2, 4, 84, 84) -> (batch*2, 4, 84, 84)
                obs = batch['obs'].view(-1, 4, 84, 84)
                actions = batch['actions'].view(-1)
                old_log_probs = batch['log_probs'].view(-1)
                advantages = batch['advantages'].view(-1)
                returns = batch['returns'].view(-1)
                
                # 新しいログ確率とエントロピーを取得
                _, new_log_probs, dist_entropy = self.agent.get_action(obs, actions)
                
                # --- Actor Loss (PPO Clipping) ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- Critic Loss (Value Function MSE) ---
                # 集中クリティックなので joint_states を使用
                v1_pred, v2_pred = self.agent.get_value(batch['joint_states'])
                # v_preds を (batch*2, 1) にまとめてリターンと比較
                v_preds = torch.cat([v1_pred, v2_pred], dim=0).squeeze()
                critic_loss = F.mse_loss(v_preds, returns)
                
                # --- Total Loss ---
                loss = actor_loss + vf_coef * critic_loss - ent_coef * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5) # 勾配爆発防止
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss

```

### クラス設計の要点

__1. `collect_rollouts`（経験の収集）__

* ここではグラフィックスメモリを節約するため `torch.no_grad()` を使用しています。
* エピソードが終了（`done`）した際に自動で `reset` し、常にデータが途切れないようにしています。

__2. `train_step`（ポリシーの更新）__

* **データのフラット化**: バッファ内では `(batch, 2, ...)` とエージェントごとに分かれていますが、学習時は `(batch*2, ...)` と一列に並べることで、1Pと2Pの区別なく「ボクシングの動き」として一気に重み共有モデル（Actor）を学習させます。
* **PPO損失計算**: 前回設計した Clipping ロジックを組み込み、古いポリシーから離れすぎないように制御しています。
* **エントロピー項**: `ent_coef` を掛けて損失から引くことで、AIが「同じ行動ばかり選ぶ」のを防ぎ、多様な動きを探索させます。

### 実行イメージ

```python
# 全体の統合
env = get_env()
agent = MAPPOAgent(action_space_n=18)
trainer = MAPPOAtariTrainer(env, agent)

# 学習ループ
for iteration in range(1000):
    trainer.collect_rollouts() # 1. データを溜める
    loss = trainer.train_step() # 2. 学習する
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {loss:.4f}")

```

これで、ボクシング攻略のための「頭脳」と「学習システム」がすべて連結されました。

## 総括

ボクシングゲーム用MAPPOの学習エージェント設計の勘所は、**「GAEでアドバンテージを安定化」「バッファで1P・2P・統合状態を完全同期」「1つのトレーナーで収集・GAE計算・PPO更新を一括管理」** の3点に集約されます。

### 1. GAE（一般化アドバンテージ推定）の勘所
- **目的**：  
  「実際の報酬（Monte Carlo）」と「1ステップ予測（TD誤差）」の**いいとこ取り**をして、アドバンテージのバラツキとバイアスをバランスさせる。
- **直感的なイメージ**：  
  パンチを打った瞬間（今の予測）だけでなく、少し先の未来（実際に当たるまで）も指数関数的に重み付けして評価する。
- **パラメータ**：
  - $\lambda$（ラムダ）で「どれだけ未来を見るか」を調整（通常 $\lambda=0.95$）。
  - $\lambda=0$ ならTD誤差だけ、$\lambda=1$ なら最後まで足し算（Monte Carlo）。
- **MAPPOでの重要性**：  
  相手の動きという不確定要素が加わるため報酬のノイズが大きい。集中クリティックの価値予測をベースにGAEを計算することで、**ノイズに強く、質の高いアドバンテージ**を得られる。

### 2. MAPPOAtariTrainer の設計勘所
- **役割**：  
  環境・ネットワーク・バッファを統括し、「データ収集 → GAE計算 → PPO更新」を一気通貫で回すコアエンジン。
- **collect_rollouts（経験収集）**：
  - `torch.no_grad()` でメモリ節約しつつ、1P・2Pの行動と価値を同時に取得。
  - 終了フラグ（`done`）で自動リセットし、バッファを途切れなく埋める。
  - 最後の状態の価値を取得し、バッファ内で**後ろから遡るGAE計算**を呼び出す。
- **train_step（PPO更新）**：
  - バッファのデータを `(batch, 2, ...)` → `(batch*2, ...)` にフラット化し、1P・2Pをまとめて学習（重み共有Actorを一括更新）。
  - Actor損失：PPOクリッピング（`ratio` を `1±clip` に制限）で古いポリシーから離れすぎないようにする。
  - Critic損失：集中クリティックの価値予測とリターンのMSE。
  - エントロピー項：`ent_coef` で探索を促進し、同じ行動ばかり選ばないようにする。
  - 勾配クリッピングで学習の安定化。

### 3. 全体としての設計思想
- **データの完全同期**：  
  1Pの観測・2Pの観測・統合状態（joint state）・行動・報酬・価値・終了フラグを、バッファの同じインデックスで管理し、GAE計算とミニバッチ生成でズレないようにする。
- **集中クリティック＋GAEの組み合わせ**：  
  神の視点（8ch統合状態）で価値を予測し、その予測をベースにGAEでアドバンテージを計算する。これにより、相手の動きを含む非定常環境でも安定した学習が可能になる。
- **1つのトレーナーで統合**：  
  環境とのインタラクション、バッファ管理、GAE計算、PPO更新を1クラスにまとめることで、ボクシング用MAPPOの学習ループをシンプルかつ堅牢に実装できる。

