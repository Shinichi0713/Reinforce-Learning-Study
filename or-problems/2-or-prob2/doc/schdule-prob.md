
問題説明文

はい、`num_jobs = 5`、`num_machines = 2` という設定を前提にした**具体的な問題定義**と、説明文を以下にまとめます。

---

## 問題の前提（具体的な定義）

- ジョブ（作業）は**5個**あります。  
  例：ジョブ0、ジョブ1、ジョブ2、ジョブ3、ジョブ4
- 各ジョブには**所要時間**（1～9の整数値、毎回ランダムに決まる）が割り当てられています。
- ジョブを処理するための**マシン（機械）は2台**あります。  
  例：マシン0、マシン1
- 各ジョブは、どちらか1台のマシンで**一度だけ**処理されます。
- 各マシンは、**一度に1つのジョブしか処理できません**。  
  （ただし、割り当て順は自由です。マシン0に2つ連続で割り当てても良い）
- 各ジョブの処理は**事前に割り当てられた順番**で進みます（途中で割り当て変更はできません）。

---

## 問題の説明文（例）

---

### スケジューリング問題の定義

5つのジョブ（作業）と2台のマシン（機械）が与えられています。  
各ジョブにはランダムに決まる所要時間（1～9）が割り当てられています。

あなたの目的は、**すべてのジョブをどの順番でどのマシンに割り当てるか**を決定し、  
**全ジョブの処理が完了するまでにかかる時間（makespan）**を最小化することです。

- 各ジョブは、どちらか1台のマシンで一度だけ処理できます。
- 各マシンは、同時に1つのジョブしか処理できません。
- ジョブの割り当て順や、どのマシンに割り当てるかは自由です。

例えば、  
- ジョブ0（所要時間5）をマシン0に割り当てる
- ジョブ1（所要時間7）をマシン1に割り当てる
- ジョブ2（所要時間2）をマシン0に割り当てる  
…というように、ジョブとマシンの割り当てを順番に決めていきます。

すべてのジョブが処理し終わった時点で、  
**最も遅く終わったマシンの終了時刻（makespan）**が、全体のスケジューリングの所要時間となります。

このmakespanをできるだけ小さくする（=全体の作業を素早く終わらせる）ことが目標です。

---




---

## 1. 環境クラス（JobShopEnv）

```python
import numpy as np

class JobShopEnv:
    def __init__(self, num_jobs=5, num_machines=2):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.job_durations = np.random.randint(1, 10, size=(num_jobs,))
        self.reset()
        
    def reset(self):
        self.machine_times = np.zeros(self.num_machines)  # 各マシンの現在の終了時刻
        self.jobs_left = list(range(self.num_jobs))
        self.done = False
        return self._get_obs()
    
    def _get_obs(self):
        # 状態：各マシンの終了時刻 + 残りジョブの所要時間
        obs = np.concatenate([self.machine_times, self.job_durations[self.jobs_left]])
        return obs
    
    def step(self, action):
        # action: (ジョブidx, マシンidx)
        job_idx, machine_idx = action
        job = self.jobs_left[job_idx]
        duration = self.job_durations[job]
        self.machine_times[machine_idx] += duration
        self.jobs_left.pop(job_idx)
        done = len(self.jobs_left) == 0
        reward = 0
        if done:
            reward = -self.machine_times.max()  # makespanのマイナス
        return self._get_obs(), reward, done, {}
```

---

## 2. ポリシーネットワーク

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_jobs, num_machines):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_jobs * num_machines)
        self.num_jobs = num_jobs
        self.num_machines = num_machines

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits.view(-1, self.num_jobs, self.num_machines)
```

---

## 3. 強化学習ループ（REINFORCE）

```python
import torch.optim as optim

def train_scheduler():
    num_jobs = 5
    num_machines = 2
    hidden_dim = 64
    env = JobShopEnv(num_jobs, num_machines)
    input_dim = num_machines + num_jobs  # 状態ベクトル長
    policy = PolicyNet(input_dim, hidden_dim, num_jobs, num_machines)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    num_episodes = 2000

    for episode in range(num_episodes):
        obs = env.reset()
        log_probs = []
        rewards = []
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_tensor)
            # 残ジョブの数だけ有効
            valid_jobs = len(env.jobs_left)
            valid_logits = logits[0, :valid_jobs, :]
            probs = F.softmax(valid_logits.flatten(), dim=0)
            m = torch.distributions.Categorical(probs)
            action_idx = m.sample()
            job_idx = action_idx // env.num_machines
            machine_idx = action_idx % env.num_machines
            log_prob = m.log_prob(action_idx)
            obs, reward, done, _ = env.step((job_idx.item(), machine_idx.item()))
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        # REINFORCE
        total_reward = sum(rewards)
        loss = -torch.stack(log_probs).sum() * total_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Makespan: {-total_reward:.2f}")

train_scheduler()
```

---

### サンプルのポイント

- **状態**: 各マシンの終了時刻＋残りジョブの所要時間
- **行動**: (残ジョブidx, マシンidx) の組み合わせ
- **報酬**: 全ジョブ終了時のmakespan（最大終了時刻）のマイナス
- **アルゴリズム**: REINFORCE（ベースラインなし）

---

このサンプルは「最小限」ですが、  
- ベースライン導入  
- 状態・行動空間の工夫  
- より複雑なジョブショップ問題  
なども拡張可能です。

以上です。