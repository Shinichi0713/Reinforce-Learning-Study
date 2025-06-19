import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class JobShopEnv:
    def __init__(self, n_jobs=5, n_machines=2):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.reset()

    # 環境リセット
    def reset(self):
        # 所要時間はランダム
        self.proc_times = np.random.randint(1, 10, size=self.n_jobs)
        # アサインされていない
        self.assigned = np.zeros(self.n_jobs, dtype=bool)
        # 各マシンの現在の空き時間
        self.machine_times = np.zeros(self.n_machines)
        self.schedule = []
        return self.proc_times.copy()

    def step(self, job, machine):
        assert not self.assigned[job]
        prev_makespan = self.machine_times.max()
        self.assigned[job] = True
        self.machine_times[machine] += self.proc_times[job]
        self.schedule.append((job, machine))
        done = self.assigned.all()
        new_makespan = self.machine_times.max()
        # 逐次報酬：makespan増加分のマイナス
        reward = -(new_makespan - prev_makespan)
        if done:
            reward += -new_makespan  # 最終的にもmakespan最小化を促す
        return self.get_state(), reward, done

    def get_state(self):
        # 各ジョブごとに [proc_time, assigned_flag] を返す
        # 各マシンの現在の空き時間も返す
        return (
            self.proc_times.copy(),             # [n_jobs]
            self.assigned.astype(np.float32),   # [n_jobs]
            self.machine_times.copy(),          # [n_machines]
        )


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
if __name__ == "__main__":
    env = JobShopEnv(n_jobs=5, n_machines=2)
    imitation_policy = imitation_learning(env)
