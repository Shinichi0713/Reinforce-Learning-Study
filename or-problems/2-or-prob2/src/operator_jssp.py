
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from agent import TransformerPolicy, Critic
from environment import JobShopEnv


# 訓練コード
def train(baseline_type='critic'):
    n_jobs, n_machines = 5, 2
    env = JobShopEnv(n_jobs, n_machines)
    policy = TransformerPolicy(n_jobs, n_machines)
    if baseline_type == 'critic':
        critic = Critic(n_jobs, n_machines)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    optimizer_actor = optim.Adam(policy.parameters(), lr=1e-4)
    num_episodes = 25000
    all_returns = []
    entropy_coef = 0.01
    moving_avg = 0
    moving_avg_beta = 0.9  # 移動平均の減衰率

    policy.train()
    if baseline_type == 'critic':
        critic.train()
    
    # ログの初期化
    loss_actor_history = []
    loss_critic_history = [] if baseline_type == 'critic' else None
    reward_history = []

    for ep in range(num_episodes):
        proc_times_np, assigned_np, machine_times_np = env.reset(), np.zeros(n_jobs), np.zeros(n_machines)
        proc_times = torch.tensor(proc_times_np, dtype=torch.float32)
        assigned = torch.tensor(assigned_np, dtype=torch.float32)
        machine_times = torch.tensor(machine_times_np, dtype=torch.float32)
        log_probs = []
        entropies = []
        states = []
        done = False

        # エピソードの開始
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
        reward = -makespan.to(policy.device)
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

        optimizer_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer_actor.step()

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
            critic.save_to_state_dict() if baseline_type == 'critic' else None

        # ログの保存
        loss_actor_history.append(loss.item())
        if baseline_type == 'critic':
            loss_critic_history.append(critic_loss.item())
        reward_history.append(reward)

    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(os.path.join(dir_current, "loss_actor_history.txt"), str(loss_actor_history))
    if baseline_type == 'critic':
        write_log(os.path.join(dir_current, "loss_critic_history.txt"), str(loss_critic_history))
    write_log(os.path.join(dir_current, "reward_history.txt"), str(reward_history))
    print("Training complete.")

# ログファイルに書き込む関数
def write_log(file_path, data):
    with open(file_path, 'a') as f:
        f.write(data + '\n')

# エージェントコード
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
    train()
    eval()
