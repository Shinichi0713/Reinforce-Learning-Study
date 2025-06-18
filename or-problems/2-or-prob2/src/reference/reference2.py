import numpy as np
import torch.optim as optim


class JobShopEnv:
    def __init__(self, n_jobs=5, n_machines=2):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.reset()

    def reset(self):
        self.proc_times = np.random.randint(1, 10, size=self.n_jobs)
        self.assigned = np.zeros(self.n_jobs, dtype=bool)
        self.machine_times = np.zeros(self.n_machines)
        self.schedule = []
        return self.proc_times.copy()

    def step(self, job, machine):
        assert not self.assigned[job]
        self.assigned[job] = True
        self.machine_times[machine] += self.proc_times[job]
        self.schedule.append((job, machine))
        done = self.assigned.all()
        reward = 0
        if done:
            reward = -self.machine_times.max()  # makespanを最小化
        return self.proc_times.copy(), reward, done, {}

    def get_state(self):
        return np.concatenate([self.proc_times, self.assigned.astype(np.float32), self.machine_times])

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPolicy(nn.Module):
    def __init__(self, n_jobs, n_machines, emb_dim=256):
        super().__init__()
        self.job_embed = nn.Embedding(10, emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
        self.machine_fc = nn.Linear(n_machines, n_machines)  # シンプルなマシン選択用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, proc_times, assigned, machine_times):
        # ジョブ選択
        job_emb = self.job_embed(proc_times.to(self.device))  # [n_jobs, emb_dim]
        scores = self.fc(job_emb.to(self.device)).squeeze(-1)  # [n_jobs]
        mask = assigned > 0
        scores = scores.masked_fill(mask.to(self.device), float('-inf'))
        job_probs = F.softmax(scores, dim=-1)
        return job_probs

    def select_machine(self, machine_times):
        # マシン選択（最も空いているマシンを選ぶ例）
        scores = -machine_times  # 小さいほど良い
        probs = F.softmax(scores, dim=-1)
        return probs





def train():
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = AttentionPolicy(n_jobs, n_machines)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    num_episodes = 9000
    all_returns = []
    gamma = 1.0  # 割引率（この問題では1.0でOK）

    for ep in range(num_episodes):
        proc_times = torch.tensor(env.reset(), dtype=torch.long)
        assigned = torch.zeros(n_jobs, dtype=torch.bool)
        machine_times = torch.zeros(n_machines)
        log_probs = []
        rewards = []
        done = False
        while not done:
            # ジョブ選択
            job_probs = policy(proc_times, assigned, machine_times)
            job_dist = torch.distributions.Categorical(job_probs)
            job = job_dist.sample()
            # マシン選択
            machine_probs = policy.select_machine(machine_times)
            machine_dist = torch.distributions.Categorical(machine_probs)
            machine = machine_dist.sample()
            # 実行
            _, reward, done, _ = env.step(job.item(), machine.item())
            assigned[job] = True
            machine_times[machine] += proc_times[job]
            log_probs.append(job_dist.log_prob(job) + machine_dist.log_prob(machine))
            rewards.append(reward if done else 0)

        # 割引累積報酬（この問題ではエピソード最後だけ報酬なので不要だが、一般的な書き方）
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        all_returns.append(returns[-1].item())

        # ベースライン（平均リターン）を引く
        baseline = np.mean(all_returns[-100:]) if len(all_returns) >= 100 else np.mean(all_returns)
        returns = returns - baseline

        # 報酬の標準化
        if len(all_returns) >= 20:
            std = np.std(all_returns[-20:])
            if std > 1e-6:
                returns = returns / std

        # Policy Gradient更新
        loss = -torch.stack(log_probs).sum() * returns[-1]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 勾配クリッピング
        optimizer.step()

        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}, makespan: {-rewards[-1]:.2f}, baseline: {-baseline:.2f}")


def eval():

    proc_times = torch.tensor(env.reset(), dtype=torch.long)
    assigned = torch.zeros(n_jobs, dtype=torch.bool)
    machine_times = torch.zeros(n_machines)
    done = False
    print(f"Job times: {proc_times.tolist()}")
    while not done:
        probs = policy(proc_times, assigned, machine_times)
        machine = torch.argmax(probs).item()
        available_jobs = (~assigned).nonzero(as_tuple=True)[0]
        job = np.random.choice(available_jobs.numpy())
        _, reward, done, _ = env.step(job, machine)
        assigned[job] = True
        machine_times[machine] += proc_times[job]
        print(f"Assign job{job} to machine{machine}, machine_times: {machine_times.tolist()}")
    print(f"最終makespan: {-reward}")


if __name__ == "__main__":
    train()
