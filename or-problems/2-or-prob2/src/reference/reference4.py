import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt


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

# --- TransformerベースのPolicyネットワーク ---
class TransformerPolicy(nn.Module):
    def __init__(self, n_jobs, n_machines, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.job_embed = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.job_out = nn.Linear(d_model, 1)
        # マシン選択用のMLP
        self.machine_selector = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, "transformer_policy.pth")
        self.__load_from_state_dict()
        self.to(self.device)

    def forward(self, proc_times, assigned, machine_times):
        proc_times = proc_times.to(self.device)
        assigned = assigned.to(self.device)
        machine_times = machine_times.to(self.device)

        # ジョブごとの特徴量 [proc_time, assigned_flag]
        features = torch.stack([proc_times.float(), assigned.float()], dim=1)  # [n_jobs, 2]
        x = self.job_embed(features)  # [n_jobs, d_model]
        x = x.unsqueeze(1)  # [n_jobs, 1, d_model]
        x = self.transformer(x)  # [n_jobs, 1, d_model]
        x = x.squeeze(1)  # [n_jobs, d_model]
        scores = self.job_out(x).squeeze(-1)  # [n_jobs]
        scores = scores.masked_fill(assigned.bool(), float('-inf'))  # 割当済みジョブをマスク
        job_probs = F.softmax(scores, dim=-1)  # [n_jobs]

        # --- マシン選択 ---
        # まずジョブをサンプリング（最大値やサンプルでも良い）
        with torch.no_grad():
            selected_job_idx = torch.multinomial(job_probs, 1).item()
        selected_job_proc_time = proc_times[selected_job_idx].item()
        # 各マシンについて、[selected_job_proc_time, machine_time]を特徴量とする
        machine_features = torch.stack([
            torch.full((self.n_machines,), selected_job_proc_time, device=self.device),  # ジョブの処理時間
            machine_times  # 各マシンの空き時間
        ], dim=1)  # [n_machines, 2]
        machine_scores = self.machine_selector(machine_features).squeeze(-1)  # [n_machines]
        machine_probs = F.softmax(machine_scores, dim=-1)

        return job_probs, machine_probs

    def save_to_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_from_state_dict(self):
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device))



class Critic(nn.Module):
    def __init__(self, n_jobs, n_machines, d_model=32):
        super().__init__()
        # 状態ベクトル: ジョブ特徴量 + 機械の空き時間
        self.job_embed = nn.Linear(2, d_model)
        self.machine_embed = nn.Linear(n_machines, d_model)
        self.fc = nn.Linear(d_model * 2, 1)

    def forward(self, proc_times, assigned, machine_times):
        # proc_times, assigned: [n_jobs]、machine_times: [n_machines]
        features = torch.stack([proc_times.float(), assigned.float()], dim=1)  # [n_jobs, 2]
        job_feat = self.job_embed(features).mean(dim=0)  # [d_model]
        machine_feat = self.machine_embed(machine_times.float().unsqueeze(0)).squeeze(0)  # [d_model]
        x = torch.cat([job_feat, machine_feat], dim=-1)  # [d_model*2]
        value = self.fc(x)
        return value.squeeze(-1)  # スカラー


def train(baseline_type='critic'):
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = TransformerPolicy(n_jobs, n_machines)
    if baseline_type == 'critic':
        critic = Critic(n_jobs, n_machines)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    num_episodes = 25000
    all_returns = []
    entropy_coef = 0.01
    moving_avg = 0
    moving_avg_beta = 0.9  # 移動平均の減衰率

    for ep in range(num_episodes):
        proc_times_np, assigned_np, machine_times_np = env.reset(), np.zeros(n_jobs), np.zeros(n_machines)
        proc_times = torch.tensor(proc_times_np, dtype=torch.float32)
        assigned = torch.tensor(assigned_np, dtype=torch.float32)
        machine_times = torch.tensor(machine_times_np, dtype=torch.float32)
        log_probs = []
        entropies = []
        states = []
        done = False

        while not done:
            job_probs, machine_probs = policy(proc_times, assigned, machine_times)
            # ジョブサンプル
            job_dist = torch.distributions.Categorical(job_probs)
            job = job_dist.sample()
            # マシンサンプル
            machine_dist = torch.distributions.Categorical(machine_probs)
            machine = machine_dist.sample()

            # 環境に反映
            (next_proc_times_np, next_assigned_np, next_machine_times_np), _, done = env.step(job.item(), machine.item())
            # 状態更新
            proc_times = torch.tensor(next_proc_times_np, dtype=torch.float32)
            assigned = torch.tensor(next_assigned_np, dtype=torch.float32)
            machine_times = torch.tensor(next_machine_times_np, dtype=torch.float32)

            # ログ確率・エントロピー
            log_prob = job_dist.log_prob(job) + machine_dist.log_prob(machine)
            entropy = job_dist.entropy() + machine_dist.entropy()
            log_probs.append(log_prob)
            entropies.append(entropy)
            states.append((proc_times, assigned, machine_times))

        makespan = env.machine_times.max()
        reward = -makespan
        rewards = [0.0] * (len(log_probs) - 1) + [reward]
        returns = torch.tensor(rewards, dtype=torch.float32)

        # --- baselineの計算 ---
        if baseline_type == 'moving_average':
            if ep == 0:
                moving_avg = reward
            else:
                moving_avg = moving_avg_beta * moving_avg + (1 - moving_avg_beta) * reward
            baseline = torch.tensor([moving_avg] * len(returns), dtype=torch.float32)
        elif baseline_type == 'critic':
            with torch.no_grad():
                state_tensors = [states[-1]]  # 最終状態のみで良い
                critic_value = critic(*state_tensors[0])
                baseline = torch.zeros(len(returns))
                baseline[-1] = critic_value.item()
        else:
            baseline = returns.mean() * torch.ones_like(returns)

        advantages = returns - baseline

        log_probs = torch.stack(log_probs).to(policy.device)
        entropies = torch.stack(entropies).to(policy.device)
        loss = -torch.sum(log_probs * advantages.to(policy.device)) - entropy_coef * entropies.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Criticの学習
        if baseline_type == 'critic':
            critic_pred = critic(*states[-1])
            critic_loss = F.mse_loss(critic_pred, torch.tensor(reward, dtype=torch.float32))
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        all_returns.append(reward)
        if (ep + 1) % 500 == 0:
            avg_makespan = -np.mean([r for r in all_returns[-100:] if r < 0])
            print(f"Episode {ep+1}, avg makespan (last 100): {avg_makespan:.2f}, loss: {loss.item():.2f}")
            policy.save_to_state_dict()



def plot_gantt(assignments, proc_times, n_machines):
    """
    assignments: List of (job, machine, start_time, end_time)
    proc_times: (n_jobs,) 各ジョブの処理時間
    n_machines: マシン数
    """
    colors = plt.cm.get_cmap('tab20', len(proc_times))
    fig, ax = plt.subplots(figsize=(10, 2 + n_machines))

    for assign in assignments:
        job, machine, start, end = assign
        ax.barh(machine, end - start, left=start, color=colors(job), edgecolor='black')
        ax.text((start + end) / 2, machine, f'Job{job}', va='center', ha='center', color='white', fontsize=10)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Job Assignment Gantt Chart')
    plt.tight_layout()
    plt.show()

def eval():
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = TransformerPolicy(n_jobs, n_machines)
    policy.eval()

    assignments = []
    with torch.no_grad():
        proc_times_np, assigned_np, machine_times_np = env.reset(), np.zeros(n_jobs), np.zeros(n_machines)
        proc_times = torch.tensor(proc_times_np, dtype=torch.float32).to(policy.device)
        assigned = torch.tensor(assigned_np, dtype=torch.float32).to(policy.device)
        machine_times = torch.tensor(machine_times_np, dtype=torch.float32).to(policy.device)
        done = False
        while not done:
            job_probs, machine_probs = policy(proc_times, assigned, machine_times)
            job = torch.max(job_probs, dim=0)[1]
            machine = torch.max(machine_probs, dim=0)[1]
            # 現在のマシンタイムが開始時刻
            start_time = machine_times_np[machine.item()]
            end_time = start_time + proc_times_np[job.item()]
            assignments.append((job.item(), machine.item(), start_time, end_time))

            (next_proc_times_np, next_assigned_np, next_machine_times_np), reward, done = env.step(job.item(), machine.item())
            proc_times = torch.tensor(next_proc_times_np, dtype=torch.float32).to(policy.device)
            assigned = torch.tensor(next_assigned_np, dtype=torch.float32).to(policy.device)
            machine_times = torch.tensor(next_machine_times_np, dtype=torch.float32).to(policy.device)
            proc_times_np, machine_times_np = next_proc_times_np, next_machine_times_np

    plot_gantt(assignments, proc_times_np, n_machines)


if __name__ == "__main__":
    # train()
    eval()