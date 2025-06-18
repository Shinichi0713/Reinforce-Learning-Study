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
        return self.get_state(), reward, done, {}

    def get_state(self):
        # 各ジョブごとに [proc_time, assigned_flag] を返す
        # 各マシンの現在の空き時間も返す
        return (
            self.proc_times.copy(),             # [n_jobs]
            self.assigned.astype(np.float32),   # [n_jobs]
            self.machine_times.copy(),          # [n_machines]
        )

# --- TransformerベースのPolicyネットワーク ---
class TransformerPolicy(nn.Module):
    def __init__(self, n_jobs, n_machines, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        # ジョブ特徴量を2次元に拡張
        self.job_embed = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.job_out = nn.Linear(d_model, 1)
        self.machine_fc = nn.Linear(n_machines, n_machines)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, "transformer_policy.pth")
        self.__load_from_state_dict()
        self.to(self.device)

    def forward(self, proc_times, assigned, machine_times):
        proc_times = proc_times.to(self.device)
        assigned = assigned.to(self.device)
        # ジョブごとの特徴量 [proc_time, assigned_flag]
        features = torch.stack([proc_times.float(), assigned.float()], dim=1)  # [n_jobs, 2]
        x = self.job_embed(features)  # [n_jobs, d_model]
        x = x.unsqueeze(1)  # [n_jobs, 1, d_model]
        x = self.transformer(x)  # [n_jobs, 1, d_model]
        x = x.squeeze(1)  # [n_jobs, d_model]
        scores = self.job_out(x).squeeze(-1)  # [n_jobs]
        scores = scores.masked_fill(assigned.bool(), float('-inf'))  # 割当済みジョブをマスク
        job_probs = F.softmax(scores, dim=-1)
        return job_probs

    def select_machine(self, machine_times):
        machine_times = machine_times.to(self.device)
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

def train():
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = TransformerPolicy(n_jobs, n_machines)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    num_episodes = 15000
    gamma = 0.99
    all_returns = []
    entropy_coef = 0.01  # エントロピー正則化

    for ep in range(num_episodes):
        proc_times_np, assigned_np, machine_times_np = env.reset(), np.zeros(n_jobs), np.zeros(n_machines)
        proc_times = torch.tensor(proc_times_np, dtype=torch.float32)
        assigned = torch.tensor(assigned_np, dtype=torch.float32)
        machine_times = torch.tensor(machine_times_np, dtype=torch.float32)
        log_probs = []
        rewards = []
        entropies = []
        done = False

        while not done:
            job_probs = policy(proc_times, assigned, machine_times)
            job_dist = torch.distributions.Categorical(job_probs)
            job = job_dist.sample()
            machine_probs = policy.select_machine(machine_times)
            machine_dist = torch.distributions.Categorical(machine_probs)
            machine = machine_dist.sample()

            (next_proc_times_np, next_assigned_np, next_machine_times_np), reward, done, _ = env.step(job.item(), machine.item())
            # 状態更新
            proc_times = torch.tensor(next_proc_times_np, dtype=torch.float32)
            assigned = torch.tensor(next_assigned_np, dtype=torch.float32)
            machine_times = torch.tensor(next_machine_times_np, dtype=torch.float32)

            log_prob = job_dist.log_prob(job) + machine_dist.log_prob(machine)
            entropy = job_dist.entropy() + machine_dist.entropy()
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

        # 割引報酬計算
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        advantages = returns - baseline

        log_probs = torch.stack(log_probs).to(policy.device)
        entropies = torch.stack(entropies).to(policy.device)
        loss = -torch.sum(log_probs * advantages.to(policy.device)) - entropy_coef * entropies.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        all_returns.append(returns[0].item())
        if (ep+1) % 500 == 0:
            avg_makespan = -np.mean([r for r in all_returns[-100:] if r < 0])
            print(f"Episode {ep+1}, avg makespan (last 100): {avg_makespan:.2f}, loss: {loss.item():.2f}")
            policy.save_to_state_dict()

if __name__ == "__main__":
    train()
