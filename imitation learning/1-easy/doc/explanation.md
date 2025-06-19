もちろんです。  
ここでは「ジョブショップスケジューリング問題」を例に、**模倣学習（Imitation Learning, Behavioral Cloning）**の基本的な実装例を示します。

---

## 概要

1. **エキスパート（教師）方策**を用意し、その行動データ（状態→行動）を収集
2. そのデータを使って、ニューラルネットワーク方策（policy）を**教師あり学習**で訓練

---

## 例：ジョブショップスケジューリングの模倣学習

### （1）エキスパート方策の例

ここでは「最短の処理時間を持つジョブを最も早く空くマシンに割り当てる」という貪欲法をエキスパートとします。

### （2）データ収集

エキスパートで複数エピソード分の状態・行動ペアを収集します。

### （3）模倣学習用ネットワークの訓練

状態→行動（ジョブ・マシン）を予測するようにクロスエントロピー損失で学習します。

---

### コード例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- 簡易エキスパート方策 ---
def expert_policy(proc_times, assigned, machine_times):
    # 割り当て可能なジョブのうち処理時間が最小のものを選ぶ
    mask = (assigned == 0)
    if not np.any(mask):
        return None, None
    available_jobs = np.where(mask)[0]
    selected_job = available_jobs[np.argmin(proc_times[available_jobs])]
    # 最も早く空くマシンを選ぶ
    selected_machine = np.argmin(machine_times)
    return selected_job, selected_machine

# --- 簡易ネットワーク（状態→ジョブidx, マシンidxを予測） ---
class ImitationPolicy(nn.Module):
    def __init__(self, n_jobs, n_machines):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_jobs*2 + n_machines, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.job_head = nn.Linear(64, n_jobs)
        self.machine_head = nn.Linear(64, n_machines)
    def forward(self, proc_times, assigned, machine_times):
        x = torch.cat([proc_times, assigned, machine_times], dim=-1)
        h = self.fc(x)
        job_logits = self.job_head(h)
        machine_logits = self.machine_head(h)
        return job_logits, machine_logits

# --- データ収集 ---
def collect_expert_data(env, num_episodes=500):
    n_jobs, n_machines = env.n_jobs, env.n_machines
    data = []
    for _ in range(num_episodes):
        proc_times_np, assigned_np, machine_times_np = env.reset(), np.zeros(n_jobs), np.zeros(n_machines)
        done = False
        while not done:
            job, machine = expert_policy(proc_times_np, assigned_np, machine_times_np)
            if job is None: break
            state = (proc_times_np.copy(), assigned_np.copy(), machine_times_np.copy())
            data.append((state, job, machine))
            (proc_times_np, assigned_np, machine_times_np), _, done = env.step(job, machine)
    return data

# --- データセット化 ---
class ImitationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        (proc_times, assigned, machine_times), job, machine = self.data[idx]
        return (
            torch.tensor(proc_times, dtype=torch.float32),
            torch.tensor(assigned, dtype=torch.float32),
            torch.tensor(machine_times, dtype=torch.float32),
            torch.tensor(job, dtype=torch.long),
            torch.tensor(machine, dtype=torch.long)
        )

# --- 模倣学習の学習ループ ---
def imitation_learning(env, num_episodes=500, epochs=10, batch_size=64):
    n_jobs, n_machines = env.n_jobs, env.n_machines
    # 1. データ収集
    data = collect_expert_data(env, num_episodes)
    dataset = ImitationDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 2. モデル定義
    policy = ImitationPolicy(n_jobs, n_machines)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    # 3. 学習
    for epoch in range(epochs):
        total_loss = 0
        for proc_times, assigned, machine_times, job, machine in dataloader:
            job_logits, machine_logits = policy(proc_times, assigned, machine_times)
            loss_job = F.cross_entropy(job_logits, job)
            loss_machine = F.cross_entropy(machine_logits, machine)
            loss = loss_job + loss_machine
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * proc_times.size(0)
        print(f"Epoch {epoch+1}, loss: {total_loss/len(dataset):.4f}")
    return policy

# --- 使い方例 ---
# env = JobShopEnv(n_jobs=5, n_machines=2)
# imitation_policy = imitation_learning(env)
```

---

#### 補足

- `JobShopEnv`はあなたの環境クラスを使ってください。
- この例では「状態→ジョブ・マシンのインデックス」を同時に予測しています。
- エキスパート方策をもっと高度なものに変えれば、そのまま模倣できます。

---

ご参考になれば幸いです。

以上です。