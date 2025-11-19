# 学習コード

目的は「学習しやすく、CTDE（集中学習・分散実行）で価値分解（VDN）を使った協調学習の動きが分かる」ことです。

---

## 概要（今回の実装の特徴）

* 言語： **Python 3** （PyTorch使用）
* 環境：小さなグリッド（デフォルト5x5）、複数エージェント（デフォルト2台）
  * 複数のピックアップ地点と1つのドロップオフ地点
  * 各エージェントの行動： `UP, RIGHT, DOWN, LEFT, WAIT, PICK/DROP`（6アクション）
  * 衝突（同じセルに複数来る）でペナルティ
  * ゴール（配達完了）で報酬
* 学習法：**VDN（Value Decomposition Network）** の簡易実装
  * 各エージェントに独立のQネットワーク（観測 → Q(a)）
  * 全体の Q_total = Σ_i Q_i をターゲットとする **集中学習**
  * 実行（行動選択）は各エージェントのローカルQに基づく（ε-greedy） → **分散実行**
* 組み込み済みの簡単な replay buffer と学習ループを含む最小実装

---

## 使い方（準備）

1. Python 3.8+ を用意
2. 必要ライブラリをインストール：

```bash
pip install torch numpy
```

3. 下記のスクリプトをファイル（例 `marl_warehouse_vdn.py`）として保存し、実行：

```bash
python marl_warehouse_vdn.py
```

---

## コード（1ファイルで動きます）

```python
"""
marl_warehouse_vdn.py
シンプルなマルチエージェント倉庫環境 + VDN 学習の最小実装

依存: torch, numpy
実行: python marl_warehouse_vdn.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# -------------------------
# Config / ハイパーパラメータ
# -------------------------
GRID_ROWS = 5
GRID_COLS = 5
N_AGENTS = 2
N_PICKUPS = 2
DROP_POS = (GRID_ROWS-1, GRID_COLS-1)
EPISODES = 1000
MAX_STEPS = 50
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
BUFFER_CAP = 10000
TARGET_UPDATE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# 環境実装（簡易）
# 状態表現: 全体状態は各エージェント位置 + 各ピックアップの残有無
# 各エージェント観測: 自分の (r,c) とピックアップとドロップ位置の情報（フラット）
# -------------------------
Action = namedtuple('Action', ['UP','RIGHT','DOWN','LEFT','WAIT','PICKDROP'])(
    0,1,2,3,4,5
)
N_ACTIONS = 6

class WarehouseEnv:
    def __init__(self, rows=GRID_ROWS, cols=GRID_COLS, n_agents=N_AGENTS, n_pickups=N_PICKUPS):
        self.rows = rows
        self.cols = cols
        self.n_agents = n_agents
        self.n_pickups = n_pickups
        self.drop = DROP_POS
        self.reset()

    def reset(self):
        # エージェントを左上隅付近にランダム配置（重なりなし）
        self.agent_pos = []
        used = set()
        for i in range(self.n_agents):
            while True:
                r = random.randint(0, self.rows//2)
                c = random.randint(0, self.cols//2)
                if (r,c) not in used:
                    used.add((r,c))
                    self.agent_pos.append((r,c))
                    break
        # ピックアップはランダムに配置（ドロップと重ならない）
        self.pickups = []
        used2 = set(self.agent_pos)
        used2.add(self.drop)
        for _ in range(self.n_pickups):
            while True:
                r = random.randint(0, self.rows-1)
                c = random.randint(0, self.cols-1)
                if (r,c) not in used2:
                    used2.add((r,c))
                    self.pickups.append(((r,c), False))  # (pos, picked_flag)
                    break
        # 各エージェントが荷を持っているか
        self.carry = [False]*self.n_agents
        self.steps = 0
        return self._get_obs(), self._get_state()

    def _get_state(self):
        # 全体状態ベクトル: agent positions flattened + pickup flags
        vec = []
        for (r,c) in self.agent_pos:
            vec.append(r/(self.rows-1))
            vec.append(c/(self.cols-1))
        for (pos, picked) in self.pickups:
            vec.append(pos[0]/(self.rows-1))
            vec.append(pos[1]/(self.cols-1))
            vec.append(1.0 if picked else 0.0)
        for carry in self.carry:
            vec.append(1.0 if carry else 0.0)
        # drop pos
        vec.append(self.drop[0]/(self.rows-1))
        vec.append(self.drop[1]/(self.cols-1))
        return np.array(vec, dtype=np.float32)

    def _get_obs(self):
        # 各エージェントの観測（ローカル観測だがここではシンプルに全体の部分情報を持たせる）
        obs_list = []
        for i in range(self.n_agents):
            r,c = self.agent_pos[i]
            v = [r/(self.rows-1), c/(self.cols-1)]
            # ピックアップ位置（全て）
            for (pos,picked) in self.pickups:
                v.append(pos[0]/(self.rows-1))
                v.append(pos[1]/(self.cols-1))
                v.append(1.0 if picked else 0.0)
            v.append(self.drop[0]/(self.rows-1))
            v.append(self.drop[1]/(self.cols-1))
            v.append(1.0 if self.carry[i] else 0.0)
            # 他エージェント相対位置（簡易）
            for j in range(self.n_agents):
                if j==i: continue
                rr,cc = self.agent_pos[j]
                v.append((rr - r)/(self.rows-1))
                v.append((cc - c)/(self.cols-1))
            obs_list.append(np.array(v, dtype=np.float32))
        return obs_list

    def step(self, actions):
        """
        actions: list of length n_agents, each action in [0..N_ACTIONS-1]
        returns: obs_list, state, rewards(list), done, info
        """
        self.steps += 1
        rewards = [0.0]*self.n_agents
        # 1) move phase: compute tentative new positions
        new_positions = list(self.agent_pos)
        for i,a in enumerate(actions):
            r,c = self.agent_pos[i]
            if a == Action.UP:
                nr, nc = max(0, r-1), c
            elif a == Action.RIGHT:
                nr, nc = r, min(self.cols-1, c+1)
            elif a == Action.DOWN:
                nr, nc = min(self.rows-1, r+1), c
            elif a == Action.LEFT:
                nr, nc = r, max(0, c-1)
            else:
                nr, nc = r, c
            new_positions[i] = (nr,nc)
        # 2) collision detection: if 2+ agents attempt same cell -> penalty and they stay
        pos_counts = {}
        for pos in new_positions:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for i,pos in enumerate(new_positions):
            if pos_counts[pos] > 1:
                # collision: heavy penalty and agent stays
                rewards[i] -= 5.0
                new_positions[i] = self.agent_pos[i]
        # 3) apply movement
        self.agent_pos = new_positions
        # 4) pick/drop handling
        # PICKDROP action picks up a pickup if at same cell and not carrying
        for i,a in enumerate(actions):
            if a == Action.PICKDROP:
                if not self.carry[i]:
                    # try to pick
                    for idx,(ppos,picked) in enumerate(self.pickups):
                        if (not picked) and (ppos == self.agent_pos[i]):
                            self.pickups[idx] = (ppos, True)
                            self.carry[i] = True
                            rewards[i] += 10.0  # pick reward small
                            break
                else:
                    # try to drop at drop location
                    if self.agent_pos[i] == self.drop:
                        self.carry[i] = False
                        rewards[i] += 50.0  # successful delivery
        # step penalty to encourage short completion
        for i in range(self.n_agents):
            rewards[i] -= 0.1
        # done when all pickups are delivered or max steps
        all_delivered = all([picked for (_,picked) in self.pickups]) and all([not c for c in self.carry])
        done = all_delivered or (self.steps >= MAX_STEPS)
        obs = self._get_obs()
        state = self._get_state()
        return obs, state, rewards, done, {}

# -------------------------
# Qネットワーク（シンプル MLP）
# -------------------------
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Replay Buffer（joint transitionを保存）
# -------------------------
Transition = namedtuple('Transition', ['obs', 'state', 'actions', 'rewards', 'next_obs', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAP):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        samples = random.sample(self.buf, batch_size)
        return samples
    def __len__(self):
        return len(self.buf)

# -------------------------
# VDN Agent： 各エージェントが独立ネットワーク、訓練は合算Qを使う
# -------------------------
class VDN:
    def __init__(self, obs_dims, n_actions, n_agents):
        self.n_agents = n_agents
        self.n_actions = n_actions
        # エージェントごとにQネットを持つ（ここでは同一アーキテクチャだが個別パラメータ）
        self.q_nets = [QNet(obs_dims[i], n_actions).to(DEVICE) for i in range(n_agents)]
        self.target_qs = [QNet(obs_dims[i], n_actions).to(DEVICE) for i in range(n_agents)]
        for i in range(n_agents):
            self.target_qs[i].load_state_dict(self.q_nets[i].state_dict())
        # オプティマイザは全パラメータまとめて
        all_params = []
        for net in self.q_nets:
            all_params += list(net.parameters())
        self.opt = optim.Adam(all_params, lr=LR)
        self.loss_fn = nn.MSELoss()

    def act(self, obs_list, epsilon):
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = self.q_nets[i](x).squeeze(0).cpu().numpy()
                if random.random() < epsilon:
                    a = random.randrange(self.n_actions)
                else:
                    a = int(np.argmax(q))
                actions.append(a)
        return actions

    def update(self, batch):
        # batch: list of Transition
        batch_size = len(batch)
        # construct tensors: for each agent, stack obs, next_obs; actions and rewards per agent
        obs_bs = [[] for _ in range(self.n_agents)]
        next_obs_bs = [[] for _ in range(self.n_agents)]
        acts = []
        rews = []
        dones = []
        for tr in batch:
            for i in range(self.n_agents):
                obs_bs[i].append(tr.obs[i])
                next_obs_bs[i].append(tr.next_obs[i])
            acts.append(tr.actions)
            rews.append(tr.rewards)
            dones.append(tr.done)
        # convert to tensors
        obs_tensors = [torch.tensor(np.array(obs_bs[i]), dtype=torch.float32, device=DEVICE) for i in range(self.n_agents)]
        next_obs_tensors = [torch.tensor(np.array(next_obs_bs[i]), dtype=torch.float32, device=DEVICE) for i in range(self.n_agents)]
        acts_t = torch.tensor(np.array(acts), dtype=torch.int64, device=DEVICE)          # shape (B, n_agents)
        rews_t = torch.tensor(np.array(rews), dtype=torch.float32, device=DEVICE)        # shape (B, n_agents)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32, device=DEVICE)      # shape (B,)
        # current Q per agent for taken actions
        q_taken = []
        q_next_max = []
        for i in range(self.n_agents):
            q_vals = self.q_nets[i](obs_tensors[i])                # (B, A)
            idx = acts_t[:, i].unsqueeze(1)                       # (B,1)
            q_sel = q_vals.gather(1, idx).squeeze(1)              # (B,)
            q_taken.append(q_sel)
            with torch.no_grad():
                qn = self.target_qs[i](next_obs_tensors[i])      # (B,A)
                qn_max = qn.max(dim=1)[0]                       # (B,)
                q_next_max.append(qn_max)
        # sum across agents: joint Q
        q_taken_joint = sum(q_taken)          # (B,)
        q_next_joint = sum(q_next_max)        # (B,)
        # compute target: r_joint + gamma * (1-done) * q_next_joint
        r_joint = rews_t.sum(dim=1)           # (B,)
        target = r_joint + (1.0 - dones_t) * GAMMA * q_next_joint
        loss = self.loss_fn(q_taken_joint, target.detach())
        # backward
        self.opt.zero_grad()
        loss.backward()
        # gradient clipping optionally
        torch.nn.utils.clip_grad_norm_(sum([list(net.parameters()) for net in self.q_nets], []), 10)
        self.opt.step()
        return loss.item()

    def update_target(self):
        for i in range(self.n_agents):
            self.target_qs[i].load_state_dict(self.q_nets[i].state_dict())

# -------------------------
# 学習ループ
# -------------------------
def train():
    env = WarehouseEnv(n_agents=N_AGENTS, n_pickups=N_PICKUPS)
    obs, state = env.reset()
    obs_dims = [len(o) for o in obs]

    agent = VDN(obs_dims, N_ACTIONS, N_AGENTS)
    buffer = ReplayBuffer()

    epsilon = EPS_START
    total_steps = 0

    for ep in range(1, EPISODES+1):
        obs, state = env.reset()
        ep_reward = 0.0
        for t in range(MAX_STEPS):
            actions = agent.act(obs, epsilon)
            next_obs, next_state, rewards, done, _ = env.step(actions)
            buffer.push(obs, state, actions, rewards, next_obs, next_state, done)
            obs = next_obs
            state = next_state
            ep_reward += sum(rewards)
            total_steps += 1

            # training step
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = agent.update(batch)

            if done:
                break
        # epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # target update
        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        if ep % 50 == 0:
            print(f"Ep {ep}  total_reward={ep_reward:.2f}  epsilon={epsilon:.3f} buffer={len(buffer)}")

    print("Training finished.")
    # show final policy by greedy rollout
    obs, state = env.reset()
    for t in range(30):
        actions = agent.act(obs, 0.0)  # greedy
        print(f"Step {t} actions: {actions} positions: {env.agent_pos} carry: {env.carry}")
        obs, state, rewards, done, _ = env.step(actions)
        if done:
            print("Done in rollout")
            break

if __name__ == "__main__":
    train()
```

---

## 説明・拡張（学習アルゴリズム面での注意）

* **この実装は学習アルゴリズムの概念実証（proof-of-concept）** です。VDN の基本的な学習ループ（局所Qの合算で全体Qを作る）を示します。
* 実際の研究やより性能を出したい場合は以下を検討してください：
  * **QMIX** （非線形に合成することで複雑なクレジット分配を学習）
  * **集中クリティック + 分散アクター（MADDPG）** （連続制御）
  * **通信学習** を導入してメッセージパッシングを学習（GNNやTransformerベース）
  * **リプレイバッファを Prioritized Replay に**
  * **観測を部分観測にして隠れ状態問題に対応** （RNNを導入）
  * **報酬シェーピング** ：局所報酬とグローバル報酬の組み合わせ調整

---

## どのように評価するか

* **学習曲線** （エピソードごとの総報酬）をプロットして学習の進行を見る
* **成功率** （すべてのピックを配達できたエピソード比）
* **衝突数や平均ステップ数** （効率と安全性の評価）

---

必要なら次を用意します（希望の番号を教えてください）：

1. QMIX（簡易）実装サンプル
2. env を部分観測に変更し、RNN（LSTM）で方策を作るサンプル
3. 可視化（学習曲線のプロット）をする実行スクリプト
4. Docker / Colab で実行できる完全ノートブック形式

どれがよいですか？
