
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, "critic.pth")
        self.__load_from_state_dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, proc_times, assigned, machine_times):
        # proc_times, assigned: [n_jobs]、machine_times: [n_machines]
        features = torch.stack([proc_times.float(), assigned.float()], dim=1).to(self.device)  # [n_jobs, 2]
        job_feat = self.job_embed(features).mean(dim=0)  # [d_model]
        machine_feat = self.machine_embed(machine_times.float().unsqueeze(0).to(self.device)).squeeze(0)  # [d_model]
        x = torch.cat([job_feat, machine_feat], dim=-1).to(self.device)  # [d_model*2]
        value = self.fc(x)
        return value.squeeze(-1)  # スカラー
    
    def save_to_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_from_state_dict(self):
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device))
        else:
            print(f"Critic model not found at {self.path_nn}, using uninitialized weights.")
