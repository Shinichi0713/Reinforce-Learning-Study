MAVEN（Multi-Agent Variational Exploration）が探索力に優れている主な理由は、**「潜在変数 z によって各エージェントの行動価値（Q値）を多様化し、探索のバリエーションを増やす」**という設計思想にあります。

---

### 1. 通常のQMIXとの違い

通常の QMIX では：

- 各エージェントは**単一の Qネットワーク**を持ち、観測から Q値を計算します。
- Mixing Network は**グローバル報酬に基づいて Q値を合成**しますが、
  各エージェントの Q値自体は「その観測に対する最適な行動」に収束しがちです。

これに対し MAVEN では：

- **潜在変数 z** を導入し、Mixing Network の入力に加えます。
- 各エージェントの Q値は「観測 + z」に依存するため、**同じ観測でも z が変われば Q値が変わる**ようになります。
- これにより、**同じ状態でも異なる行動を選ぶ「多様なポリシー」**を学習できます。

---

### 2. 潜在変数 z が探索を促進する仕組み

MAVEN の Mixing Network は以下のように定義されます（簡略化）：

```python
class MixingNetwork(nn.Module):
    def forward(self, agent_qs, z):
        x = torch.cat([agent_qs, z], dim=-1)
        return self.net(x)
```

- `agent_qs`：各エージェントの Q値（観測に基づく）
- `z`：潜在変数（例：一様分布やガウス分布からサンプリング）

**z を変えると、同じ観測でも Q_tot（共同Q値）が変わる**ため：

- 学習中に **z をランダムにサンプリング**することで、同じ状態でも異なる行動を選ぶ「多様な探索軌跡」が生まれます。
- これにより、**局所最適に陥りにくく、より良い協調戦略を発見しやすくなります**。

---

### 3. 探索の多様性がもたらすメリット

1. **局所最適の回避**

   - 通常の QMIX では、初期の探索が偏ると特定の戦略に収束しがちです。
   - MAVEN は z によって Q値を揺らすため、**異なる戦略を試す機会が増え**、局所最適を避けやすくなります。
2. **協調タスクでの「役割分担」の発見**

   - 協調ナビゲーションのようなタスクでは、「どちらが先にボトルネックを通るか」「どちらが待機するか」といった役割分担が重要です。
   - z によって Q値が多様化すると、**異なる役割分担の戦略**を試しやすくなり、
     より良い協調戦略を発見しやすくなります。
3. **探索のバリエーション増加**

   - 通常の ε-greedy 探索では、ランダム行動は「完全にランダム」です。
   - MAVEN では、**z によって「構造化されたランダム探索」**が行われ、
     無駄な探索を減らしつつ多様性を保てます。

---

### 4. まとめ

MAVEN が探索力に優れている理由は、**潜在変数 z を導入することで、同じ観測でも異なる Q値を生成し、多様な探索軌跡を生み出す**ためです。これにより：

- 局所最適に陥りにくい
- 協調タスクでの役割分担を発見しやすい
- 探索のバリエーションが増える

といったメリットが得られます。

参考：MAVEN の元論文では、**「z によって行動価値の分布を多様化し、探索を促進する」**ことが主な貢献として述べられています。

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

import random

from collections import deque

import matplotlib.pyplot as plt

import io

from PIL import Image

from typing import List, Tuple, Dict, Optional

import copy

# --- MAVEN風のニューラルネットワーク ---

classQNetwork(nn.Module):

    """各エージェント用のQネットワーク（観測 → Q値）"""

    def__init__(self, obs_dim, action_dim, hidden_dim=64):

    super().__init__()

    self.net = nn.Sequential(

    nn.Linear(obs_dim, hidden_dim),

    nn.ReLU(),

    nn.Linear(hidden_dim, hidden_dim),

    nn.ReLU(),

    nn.Linear(hidden_dim, action_dim)

    )

    defforward(self, obs):

    returnself.net(obs)

classMixingNetwork(nn.Module):

    """

    MAVEN風のMixing Network（簡易版）

    - 入力: 各エージェントのQ値 + latent variable z

    - 出力: 共同Q値 Q_tot

    """

    def__init__(self, n_agents, hidden_dim=64, z_dim=4):

    super().__init__()

    self.n_agents = n_agents

    self.z_dim = z_dim

    self.net = nn.Sequential(

    nn.Linear(n_agents + z_dim, hidden_dim),

    nn.ReLU(),

    nn.Linear(hidden_dim, hidden_dim),

    nn.ReLU(),

    nn.Linear(hidden_dim, 1)

    )

    defforward(self, agent_qs, z):

    # agent_qs: (batch, n_agents)

    # z: (batch, z_dim)

    x = torch.cat([agent_qs, z], dim=-1)

    returnself.net(x)  # (batch, 1)

# --- 経験再生バッファ ---

class ReplayBuffer:

    def__init__(self, capacity=10000):

    self.buffer = deque(maxlen=capacity)

    defpush(self, obs_list, actions, rewards, next_obs_list, done, z):

    # z も一緒に保存

    self.buffer.append((obs_list, actions, rewards, next_obs_list, done, z))

    defsample(self, batch_size):

    batch = random.sample(self.buffer, batch_size)

    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, z_batch = zip(*batch)

    return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, z_batch

    def__len__(self):

    returnlen(self.buffer)

# --- MAVEN風トレーナー（CooperativeNavigationEnv 用） ---

class MavenTrainer:

    def__init__(self, n_agents, obs_dim, action_dim, lr=1e-3, gamma=0.99,

    hidden_dim=64, z_dim=4, target_update_interval=100, tau=0.01):

    self.n_agents = n_agents

    self.obs_dim = obs_dim

    self.action_dim = action_dim

    self.gamma = gamma

    self.z_dim = z_dim

    self.target_update_interval = target_update_interval

    self.tau = tau

    self.update_count = 0

    # 各エージェントのQネットワーク

    self.q_nets = [QNetwork(obs_dim, action_dim, hidden_dim) for _ inrange(n_agents)]

    self.target_q_nets = [QNetwork(obs_dim, action_dim, hidden_dim) for _ inrange(n_agents)]

    for i inrange(n_agents):

    self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())

    # Mixing Network

    self.mixing_net = MixingNetwork(n_agents, hidden_dim, z_dim)

    self.target_mixing_net = MixingNetwork(n_agents, hidden_dim, z_dim)

    self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    # オプティマイザ

    all_params = list(self.mixing_net.parameters())

    for q_net inself.q_nets:

    all_params.extend(list(q_net.parameters()))

    self.optimizer = optim.Adam(all_params, lr=lr)

    defsample_z(self, batch_size):

    # 一様分布に変更（探索の多様性を高める）

    return torch.rand(batch_size, self.z_dim) * 2 - 1  # [-1, 1]

    defcompute_q_tot(self, obs_batch, action_batch, z, nets, mixing_net):

    batch_size = len(obs_batch)

    agent_qs = []

    for i inrange(self.n_agents):

    obs_i = torch.FloatTensor([obs[i] for obs in obs_batch])

    actions_i = torch.LongTensor([a[i] for a in action_batch])

    q_values = nets[i](obs_i)

    q_i = q_values.gather(1, actions_i.unsqueeze(1)).squeeze(1)

    agent_qs.append(q_i)

    agent_qs = torch.stack(agent_qs, dim=1)

    q_tot = mixing_net(agent_qs, z)

    return q_tot.squeeze(1)

    defupdate(self, batch_size, buffer: ReplayBuffer):

    iflen(buffer) < batch_size:

    return

    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, z_batch = buffer.sample(batch_size)

    batch_size = len(obs_batch)

    # サンプルされた z をそのまま使う（エピソードごとに固定された z）

    z = torch.stack(z_batch, dim=0)  # (batch_size, z_dim)

    # 現在のQ_tot

    current_q_tot = self.compute_q_tot(obs_batch, action_batch, z, self.q_nets, self.mixing_net)

    # ターゲットQ_tot（同じ z を使う）

    with torch.no_grad():

    next_agent_qs = []

    for i inrange(self.n_agents):

    next_obs_i = torch.FloatTensor([next_obs[i] for next_obs in next_obs_batch])

    next_q_values = self.target_q_nets[i](next_obs_i)

    next_max_q = next_q_values.max(1)[0]

    next_agent_qs.append(next_max_q)

    next_agent_qs = torch.stack(next_agent_qs, dim=1)

    next_q_tot = self.target_mixing_net(next_agent_qs, z)  # 同じ z を使う

    rewards = torch.FloatTensor([sum(r) for r in reward_batch])

    dones = torch.FloatTensor(done_batch)

    target_q_tot = rewards + (1 - dones) * self.gamma * next_q_tot.squeeze(1)

    # TD誤差（Huber loss）

    loss = nn.SmoothL1Loss()(current_q_tot, target_q_tot)

    self.optimizer.zero_grad()

    loss.backward()

    self.optimizer.step()

    # ソフトアップデート（毎回少しずつターゲットを更新）

    self.update_count += 1

    for i inrange(self.n_agents):

    for target_param, param inzip(self.target_q_nets[i].parameters(), self.q_nets[i].parameters()):

    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    for target_param, param inzip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):

    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    return loss.item()

    defselect_actions(self, obs_list, epsilon=0.1):

    actions = []

    for i inrange(self.n_agents):

    if np.random.rand() < epsilon:

    action = np.random.randint(self.action_dim)

    else:

    obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0)

    with torch.no_grad():

    q_values = self.q_nets[i](obs_tensor)

    action = q_values.argmax().item()

    actions.append(action)

    return actions

# --- 学習ループ（修正版） ---

deftrain_maven(env, trainer, buffer, episodes=500, batch_size=32,

    epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.998):

    epsilon = epsilon_start

    rewards_history = []

    for ep inrange(episodes):

    # エピソードごとに z をサンプリングして固定

    z_ep = trainer.sample_z(1).squeeze(0)  # (z_dim,)

    obs_list = env.reset()

    done = False

    total_reward = 0.0

    step_count = 0

    # エピソード内の経験を収集（z_ep を記録）

    whilenot done and step_count < env.max_steps:

    actions = trainer.select_actions(obs_list, epsilon=epsilon)

    next_obs_list, rewards, done, info = env.step(actions)

    # z_ep を経験に含める（後で update で使う）

    buffer.push(obs_list, actions, rewards, next_obs_list, done, z_ep)

    obs_list = next_obs_list

    total_reward += sum(rewards)

    step_count += 1

    # エピソード終了後にまとめて更新

    loss = trainer.update(batch_size, buffer)

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    rewards_history.append(total_reward)

    if ep % 50 == 0:

    print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return rewards_history

# --- GIF保存（修正版） ---

defsave_gif_maven(env, trainer, filename="coop_nav_maven.gif", max_steps=30):

    # 環境をコピーして、学習ループの最後の状態を保持

    eval_env = copy.deepcopy(env)

    frames = []

    obs_list = eval_env.reset()

    done = False

    step_count = 0

    fig, ax = plt.subplots(figsize=(5, 5))

    whilenot done and step_count < max_steps:

    eval_env.render(ax)

    buf = io.BytesIO()

    plt.savefig(buf, format='png', bbox_inches='tight')

    buf.seek(0)

    frames.append(Image.open(buf))

    actions = trainer.select_actions(obs_list, epsilon=0.0)  # 評価時はε=0

    obs_list, rewards, done, info = eval_env.step(actions)

    step_count += 1

    if frames:

    frames[0].save(

    filename,

    save_all=True,

    append_images=frames[1:],

    duration=300,

    loop=0

    )

    plt.close(fig)

    print(f"✅ GIF saved as {filename}")

    else:

    plt.close(fig)

    print("❌ No frames to save")

# --- 実行例（修正版） ---

if__name__ == "__main__":

    env = CooperativeNavigationEnv(size=5)

    trainer = MavenTrainer(

    n_agents=2,

    obs_dim=8,  # _get_obs の出力次元（8次元）

    action_dim=4,  # 0〜3 の4種類（上下左右）

    lr=1e-3,

    gamma=0.99,

    hidden_dim=64,

    z_dim=4,

    target_update_interval=100,

    tau=0.01

    )

    buffer = ReplayBuffer(capacity=10000)

    # 学習（エピソード数を増加）

    rewards_history = train_maven(env, trainer, buffer, episodes=1, batch_size=32)

    # 学習曲線をプロット

    plt.plot(rewards_history)

    plt.xlabel("Episode")

    plt.ylabel("Total Reward")

    plt.title("MAVEN Training Progress (Cooperative Navigation)")

    plt.grid(True)

    plt.show()

    # 学習済みポリシーでGIFを保存

    save_gif_maven(env, trainer, filename="coop_nav_maven.gif", max_steps=30)
