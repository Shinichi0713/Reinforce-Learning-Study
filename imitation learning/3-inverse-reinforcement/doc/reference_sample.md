はい、逆強化学習（Inverse Reinforcement Learning, IRL）のシンプルなサンプルコードをPython（PyTorch使用）でご紹介します。  
ここでは代表的な「最大エントロピー逆強化学習（MaxEnt IRL）」の考え方を用い、GridWorld環境を例にします。

---

## 1. 必要なライブラリのインストール

```bash
pip install numpy torch matplotlib
```

---

## 2. サンプルコード（MaxEnt IRL, GridWorld）

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- GridWorld環境の定義 ---
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right

    def state_to_xy(self, s):
        return s // self.size, s % self.size

    def xy_to_state(self, x, y):
        return x * self.size + y

    def step(self, s, a):
        x, y = self.state_to_xy(s)
        if a == 0 and x > 0: x -= 1
        if a == 1 and x < self.size - 1: x += 1
        if a == 2 and y > 0: y -= 1
        if a == 3 and y < self.size - 1: y += 1
        return self.xy_to_state(x, y)

# --- エキスパートの軌跡を生成 ---
def generate_expert_trajectories(env, goal_state, n_trajs=20):
    trajectories = []
    for _ in range(n_trajs):
        traj = []
        s = np.random.randint(env.n_states)
        for _ in range(10):
            x, y = env.state_to_xy(s)
            gx, gy = env.state_to_xy(goal_state)
            if abs(gx - x) > abs(gy - y):
                a = 1 if gx > x else 0
            else:
                a = 3 if gy > y else 2
            traj.append((s, a))
            s = env.step(s, a)
        trajectories.append(traj)
    return trajectories

# --- 報酬関数のパラメータ化 ---
class RewardNet(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(n_states))

    def forward(self, states):
        return self.r[states]

# --- MaxEnt IRL の主要部分 ---
def maxent_irl(env, expert_trajs, lr=0.1, n_iters=100):
    reward_net = RewardNet(env.n_states)
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    # エキスパートの状態訪問頻度
    expert_counts = np.zeros(env.n_states)
    for traj in expert_trajs:
        for (s, a) in traj:
            expert_counts[s] += 1
    expert_counts /= expert_counts.sum()

    for it in range(n_iters):
        # 報酬を取得
        rewards = reward_net.r.detach().numpy()
        # 状態価値関数の近似（soft value iteration）
        V = np.zeros(env.n_states)
        for _ in range(100):
            V_new = np.zeros_like(V)
            for s in range(env.n_states):
                vals = []
                for a in range(env.n_actions):
                    s_ = env.step(s, a)
                    vals.append(rewards[s_] + V[s_])
                V_new[s] = np.log(np.sum(np.exp(vals)))
            V = V_new

        # 状態訪問頻度の計算（soft policy下で）
        state_counts = np.zeros(env.n_states)
        for _ in range(100):
            s = np.random.randint(env.n_states)
            for _ in range(10):
                # soft policy
                vals = []
                for a in range(env.n_actions):
                    s_ = env.step(s, a)
                    vals.append(rewards[s_] + V[s_])
                probs = np.exp(vals - np.max(vals))
                probs /= probs.sum()
                a = np.random.choice(env.n_actions, p=probs)
                state_counts[s] += 1
                s = env.step(s, a)
        state_counts /= state_counts.sum()

        # 損失 = エキスパート訪問頻度とモデル訪問頻度の差
        loss = -torch.dot(reward_net.r, torch.tensor(expert_counts - state_counts, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f"Iter {it}, Loss: {loss.item():.4f}")

    return reward_net.r.detach().numpy()

# --- 実行例 ---
env = GridWorld(size=5)
goal_state = env.xy_to_state(4, 4)
expert_trajs = generate_expert_trajectories(env, goal_state, n_trajs=50)
rewards = maxent_irl(env, expert_trajs, lr=0.05, n_iters=100)

# --- 結果の可視化 ---
plt.imshow(rewards.reshape(5, 5))
plt.colorbar()
plt.title("Recovered Reward")
plt.show()
```

---

### ポイント
- **GridWorld**での逆強化学習（MaxEnt IRL）の超簡易実装例です。
- エキスパート軌跡を模倣しつつ、「報酬関数そのもの」を推定します。
- 報酬関数は状態ごとにパラメータ化（線形）しています。
- 学習が進むと、ゴール付近が高報酬になるはずです。

---

もしもっと本格的な環境や深層学習ベースのIRL、あるいはOpenAI Gym環境でのIRLに興味があれば、`imitation`や`stable-baselines3`のライブラリもご紹介できますのでお知らせください。

以上です。