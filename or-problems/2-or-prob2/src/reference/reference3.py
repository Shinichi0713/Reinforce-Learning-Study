
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


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


# --- TransformerベースのPolicyネットワーク ---
class TransformerPolicy(nn.Module):
    def __init__(self, n_jobs, n_machines, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        # ジョブ特徴量を埋め込み
        self.job_embed = nn.Linear(1, d_model)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # ジョブ選択用の出力層
        self.job_out = nn.Linear(d_model, 1)
        # マシン選択用
        self.machine_fc = nn.Linear(n_machines, n_machines)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, "transformer_policy.pth")

        self.__load_from_state_dict()
        self.to(self.device)

    def forward(self, proc_times, assigned, machine_times):
        # proc_times: [n_jobs] (例: [7, 3, 2, 5, 4])
        # assigned: [n_jobs] (bool)
        # machine_times: [n_machines]
        proc_times = proc_times.to(self.device)
        assigned = assigned.to(self.device)
        machine_times = machine_times.to(self.device)
        x = proc_times.unsqueeze(-1).float()  # [n_jobs, 1]
        x = self.job_embed(x)  # [n_jobs, d_model]
        x = x.unsqueeze(1)  # [n_jobs, 1, d_model] (Transformerは[S, N, E])
        x = self.transformer(x)  # [n_jobs, 1, d_model]
        x = x.squeeze(1)  # [n_jobs, d_model]
        scores = self.job_out(x).squeeze(-1)  # [n_jobs]
        # 割り当て済ジョブにはマスク
        scores = scores.masked_fill(assigned, float('-inf'))
        job_probs = F.softmax(scores, dim=-1)
        return job_probs

    def select_machine(self, machine_times):
        # 最も空いているマシンを選ぶ例
        scores = -machine_times  # 小さいほど良い
        probs = F.softmax(scores, dim=-1)
        return probs
    
    def save_to_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_from_state_dict(self):
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device))

# --- 学習ループ ---
def train():
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = TransformerPolicy(n_jobs, n_machines)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    num_episodes = 15000
    all_returns = []

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
            # assignedをコピーして更新
            assigned = assigned.clone()
            assigned[job] = True
            machine_times = machine_times.clone()
            machine_times[machine] += proc_times[job]
            log_probs.append(job_dist.log_prob(job) + machine_dist.log_prob(machine))
            rewards.append(reward if done else 0)

        # REINFORCE
        total_reward = rewards[-1]
        all_returns.append(total_reward)
        baseline = sum(all_returns[-100:]) / min(len(all_returns), 100)
        advantage = total_reward - baseline
        loss = -torch.stack(log_probs).sum() * advantage

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}, makespan: {-total_reward:.2f}, baseline: {-baseline:.2f}")
            policy.save_to_state_dict()


if __name__ == "__main__":
    train()