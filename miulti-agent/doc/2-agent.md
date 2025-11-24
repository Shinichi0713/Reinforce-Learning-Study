**最小限の QMIX 学習エージェント実装（PyTorch）** を示します。

設計方針は「 **分かりやすさ優先の実証実装（教育用）** 」で、QMIX の要点（各エージェントの Q を中央の Mixer で合成し、CTDE で学習／分散実行）を含んでいます。

---

## 使い方

1. 既に提示済みの `WarehouseEnv` を同じファイルかモジュールとして用意してください（そのまま一緒の .py に貼り付けても良いです）。
2. 以下コードを保存（例 `qmix_train.py`）。
3. 必要ライブラリを入れる：

```bash
pip install torch numpy matplotlib
```

4. 実行：

```bash
python qmix_train.py
```

---

## 実装（そのまま実行できる1ファイル）

（`WarehouseEnv` を同一ファイル上部に置くか、インポートできる状態で実行してください）

```python
# qmix_train.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import time

# ---------- ハイパーパラメータ ----------
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_AGENTS = 2           # 環境と合わせる（env.num_agents）
N_ACTIONS = 5          # env.action_space (0..4)
OBS_DIM = None         # 後で env.reset() から決定
STATE_DIM = None       # 後で env._get_state() の長さから決定

EPISODES = 800
MAX_STEPS = 40
BATCH_SIZE = 32
BUFFER_CAP = 5000
GAMMA = 0.99
LR = 5e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 200   # 学習ステップごとのターゲット更新
TRAIN_START = 1000         # バッファがこれだけ貯まったら学習開始
TRAIN_FREQ = 1             # 何ステップごとに学習するか

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Replay Buffer ----------
Transition = namedtuple('Transition', ['obs', 'state', 'actions', 'rewards', 'next_obs', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, cap=BUFFER_CAP):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        return batch
    def __len__(self): return len(self.buf)

# ---------- 各エージェントのQネットワーク ----------
class AgentQNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)  # (batch, n_actions)

# ---------- Mixer（QMIX の Hypernetwork 実装：2層） ----------
class Mixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # hypernetwork for first layer weights and biases
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim * n_agents))
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim, embed_dim))

        # hypernetwork for second layer (to scalar)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, 1))

        # apply non-negative weight constraint via absolute
        self.elu = nn.ELU()

    def forward(self, agent_qs, states):
        """
        agent_qs: (batch, n_agents)  - Q values per agent (for chosen actions OR max as appropriate)
        states:   (batch, state_dim)
        returns:  (batch, 1) total Q
        """
        bs = agent_qs.size(0)
        # first layer
        w1 = torch.abs(self.hyper_w1(states))  # (bs, embed_dim * n_agents)
        b1 = self.hyper_b1(states)              # (bs, embed_dim)

        w1 = w1.view(bs, self.n_agents, self.embed_dim)  # (bs, n_agents, embed_dim)
        # agent_qs -> (bs, n_agents, 1)
        agent_qs_ = agent_qs.unsqueeze(2)
        # weighted sum: (bs, embed_dim) = sum_i agent_q_i * w1_i
        hidden = torch.bmm(agent_qs_.transpose(1,2), w1).squeeze(1)  # (bs, embed_dim)
        hidden = hidden + b1
        hidden = torch.relu(hidden)

        # second layer
        w2 = torch.abs(self.hyper_w2(states))  # (bs, embed_dim)
        b2 = self.hyper_b2(states)             # (bs,1)
        # (bs,1) = sum(hidden * w2) + b2
        y = (hidden * w2).sum(dim=1, keepdim=True) + b2
        return y  # (bs,1)

# ---------- QMIX 全体ラッパー ----------
class QMIX:
    def __init__(self, obs_dims, state_dim, n_agents=N_AGENTS, n_actions=N_ACTIONS):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dims = obs_dims
        self.state_dim = state_dim

        # agent networks and targets
        self.agent_nets = [AgentQNet(obs_dims[i], n_actions).to(DEVICE) for i in range(n_agents)]
        self.target_agent_nets = [AgentQNet(obs_dims[i], n_actions).to(DEVICE) for i in range(n_agents)]
        for i in range(n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())

        # mixer and target mixer
        self.mixer = Mixer(n_agents, state_dim).to(DEVICE)
        self.target_mixer = Mixer(n_agents, state_dim).to(DEVICE)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # optimizer over all agent nets + mixer
        params = []
        for net in self.agent_nets:
            params += list(net.parameters())
        params += list(self.mixer.parameters())
        self.opt = optim.Adam(params, lr=LR)
        self.loss_fn = nn.MSELoss()

    def select_actions(self, obs_list, epsilon):
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.tensor(obs_list[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,obs_dim)
                q = self.agent_nets[i](obs).squeeze(0).cpu().numpy()  # (n_actions,)
                if random.random() < epsilon:
                    a = random.randrange(self.n_actions)
                else:
                    a = int(np.argmax(q))
                actions.append(a)
        return actions

    def train(self, batch):
        """
        batch: list of Transition
        """
        bs = len(batch)
        # prepare tensors
        obs_b = [[] for _ in range(self.n_agents)]
        next_obs_b = [[] for _ in range(self.n_agents)]
        actions_b = []
        rewards_b = []
        states_b = []
        next_states_b = []
        dones_b = []

        for tr in batch:
            for i in range(self.n_agents):
                obs_b[i].append(tr.obs[i])
                next_obs_b[i].append(tr.next_obs[i])
            actions_b.append(tr.actions)
            rewards_b.append(tr.rewards)
            states_b.append(tr.state)
            next_states_b.append(tr.next_state)
            dones_b.append(float(tr.done))

        # tensors
        obs_t = [torch.tensor(np.array(obs_b[i]), dtype=torch.float32, device=DEVICE) for i in range(self.n_agents)]
        next_obs_t = [torch.tensor(np.array(next_obs_b[i]), dtype=torch.float32, device=DEVICE) for i in range(self.n_agents)]
        actions_t = torch.tensor(np.array(actions_b), dtype=torch.long, device=DEVICE)    # (bs, n_agents)
        rewards_t = torch.tensor(np.array(rewards_b), dtype=torch.float32, device=DEVICE) # (bs, n_agents)
        states_t = torch.tensor(np.array(states_b), dtype=torch.float32, device=DEVICE)   # (bs, state_dim)
        next_states_t = torch.tensor(np.array(next_states_b), dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(np.array(dones_b), dtype=torch.float32, device=DEVICE)     # (bs,)

        # 1) current Q for taken actions (per agent)
        q_taken = []
        for i in range(self.n_agents):
            q_vals = self.agent_nets[i](obs_t[i])          # (bs, A)
            a_idx = actions_t[:, i].unsqueeze(1)          # (bs,1)
            q_sel = q_vals.gather(1, a_idx).squeeze(1)    # (bs,)
            q_taken.append(q_sel)
        # stack to (bs, n_agents)
        q_taken_joint = torch.stack(q_taken, dim=1)       # (bs, n_agents)

        # 2) target: get target agent max Q' per agent and mix
        q_next_max = []
        with torch.no_grad():
            for i in range(self.n_agents):
                qn = self.target_agent_nets[i](next_obs_t[i])   # (bs, A)
                qn_max = qn.max(dim=1)[0]                       # (bs,)
                q_next_max.append(qn_max)
            q_next_joint = torch.stack(q_next_max, dim=1)      # (bs, n_agents)

            # compute target Q_total
            target_q_total = self.target_mixer(q_next_joint, next_states_t).squeeze(1)  # (bs,)
            # r_joint = sum rewards across agents (team reward)
            r_joint = rewards_t.sum(dim=1)                    # (bs,)
            target = r_joint + (1.0 - dones_t) * GAMMA * target_q_total  # (bs,)

        # 3) current total Q via mixer on q_taken_joint
        q_total = self.mixer(q_taken_joint, states_t).squeeze(1)  # (bs,)

        loss = self.loss_fn(q_total, target.detach())

        # optimize
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._all_parameters(), 10)
        self.opt.step()

        return loss.item()

    def _all_parameters(self):
        params = []
        for net in self.agent_nets:
            params += list(net.parameters())
        params += list(self.mixer.parameters())
        return params

    def update_target(self):
        for i in range(self.n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

# ---------- Training Loop ----------
def train_qmix(env):
    global OBS_DIM, STATE_DIM
    # get dims from env
    obs, state = env.reset()
    obs_list = list(obs.values())
    OBS_DIM = [len(o) for o in obs_list]  # each agent obs dimension
    STATE_DIM = len(state)

    print("OBS_DIM:", OBS_DIM, "STATE_DIM:", STATE_DIM)

    agent = QMIX(OBS_DIM, STATE_DIM, n_agents=env.num_agents, n_actions=env.action_space)
    buffer = ReplayBuffer()

    epsilon = EPS_START
    total_steps = 0
    losses = []

    for ep in range(1, EPISODES+1):
        ep_reward = 0.0
        obs, state = env.reset()
        for step in range(MAX_STEPS):
            obs_list = [obs[i] for i in range(env.num_agents)]
            # select actions
            actions = agent.select_actions(obs_list, epsilon)
            # step env
            next_obs, rewards, done, info = env.step({i: actions[i] for i in range(env.num_agents)})
            next_obs_list = [next_obs[i] for i in range(env.num_agents)]
            # push to buffer (convert rewards to list)
            rewards_list = [rewards[i] for i in range(env.num_agents)]
            buffer.push(obs_list, state, actions, rewards_list, next_obs_list, env._get_state(), any(done.values()))
            obs = next_obs
            state = env._get_state()
            ep_reward += sum(rewards_list)
            total_steps += 1

            # learning step
            if len(buffer) >= TRAIN_START and total_steps % TRAIN_FREQ == 0:
                batch = buffer.sample(BATCH_SIZE)
                loss = agent.train(batch)
                losses.append(loss)

            # update target networks periodically
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            if all(done.values()):
                break

        # epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % 20 == 0:
            avg_loss = np.mean(losses[-200:]) if len(losses)>0 else 0.0
            print(f"EP {ep}  reward={ep_reward:.2f}  eps={epsilon:.3f}  buf={len(buffer)}  loss={avg_loss:.4f}")

    print("Training finished.")
    return agent

# ---------- Main ----------
if __name__ == "__main__":
    # 注意: あなたの WarehouseEnv をモジュール/同一ファイル上に置いてから実行してください
    from __main__ import WarehouseEnv  # if same file, otherwise adjust import
    env = WarehouseEnv(size=10, num_agents=N_AGENTS)
    trained_agent = train_qmix(env)

    # 単独の評価ロールアウト（greedy）
    obs, state = env.reset()
    for t in range(50):
        obs_list = [obs[i] for i in range(env.num_agents)]
        actions = trained_agent.select_actions(obs_list, epsilon=0.0)  # greedy
        obs, rewards, done, info = env.step({i: actions[i] for i in range(env.num_agents)})
        env.render(mode='text')
        time.sleep(0.2)
        if all(done.values()):
            print("Done.")
            break
```

---

## 実装上の注記（重要）

1. **観測と状態**
   * 本実装では `env._get_state()` を全体状態（state）として利用しています。実際の部分観測タスクでは state が得られない環境もありますが、QMIX は *学習時に中央で state を使う* 前提（CTDE）なのでこれで問題ありません。
2. **ターゲットの計算**
   * ここでは target agent net で `max_a' Q_i(next_obs)` を取り、agent-wise に合算 → target mixer に通して `Q_total'` を得ています（標準的な近似）。
3. **単調性の担保**
   * Mixer 内で重みを `abs()` にして **非負化**しています。これにより `Q_total` が各 `Q_i` に対して単調（∂Q_total/∂Q_i >= 0）となり、QMIX の性質を満たします。
4. **簡素化点**
   * 経験の時間系列（エピソード長にまたがるバッチ）ではなく、**遷移（step）単位でサンプリング**する最も単純な実装です。QMIX 本来は時間依存の LSTM などを入れて部分観測に対応させることが多いです。
   * 優先度付きリプレイ（PER）、Double-Q、n-step などの改善は実装していません（必要に応じて追加してください）。
5. **動作確認**
   * 最初は `EPISODES` を小さくして（例 100）動作確認 → 学習性を試す場合は 500〜2000 くらいに増やしてください。

---

## 次にできる拡張（提案）

* 観測を部分観測化して **RNN（LSTM）** を agent ネットワークに導入する（COMA/QMIXでよくやる）
* **QMIX の公式実装に近づける** ：n-step TD、優先経験、ε-greedyの線形減衰、ターゲットソフトアップデートなど
* **可視化** ：学習曲線（平均報酬、成功率、衝突数）をログしてプロット
* **環境の複雑化** ：障害物、可変オーダー生成、複数ドロップオフ など

---

もしよければ、上の実装をあなたのローカルで動くように **WarehouseEnv と同一ファイルに統合した完全版** や、 **学習曲線プロット用コード（Matplotlib）** 、あるいは **QMIX を QMIX-paper 準拠に近づけた改良実装（n-step, PER）** を次に出します。どれを作りましょうか？
