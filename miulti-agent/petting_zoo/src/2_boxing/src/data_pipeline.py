import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pettingzoo.atari import boxing_v2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import supersuit as ss

def preprocess_joint_obs(obs_dict, device="cpu"):
    """
    1Pと2Pの個別の観測(84, 84, 4)を(4, 84, 84)に変換し、
    それらを結合して集中クリティック用の状態(8, 84, 84)を作る
    """
    # 1. テンソル化 (この時点では [84, 84, 4])
    t1 = torch.as_tensor(obs_dict['first_0'], dtype=torch.float32, device=device)
    t2 = torch.as_tensor(obs_dict['second_0'], dtype=torch.float32, device=device)

    # 2. 軸の入れ替え [H, W, C] -> [C, H, W] に変換
    # これにより PyTorch の Conv2d が扱える形状 [4, 84, 84] になる
    obs_1p = t1.permute(2, 0, 1) / 255.0
    obs_2p = t2.permute(2, 0, 1) / 255.0

    # 3. チャンネル方向(dim=0)に結合して 8チャンネルにする
    # 形状: (8, 84, 84)
    joint_state = torch.cat([obs_1p, obs_2p], dim=0)
    
    return obs_1p, obs_2p, joint_state

def visualize_check(joint_s):
    """統合された状態を可視化して確認する"""
    imgs = joint_s.cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    fig.suptitle("Debug: Joint Observation (Top: 1P, Bottom: 2P)", fontsize=14)
    
    for i in range(4):
        # 1Pのフレームを表示
        axes[0, i].imshow(imgs[i], cmap='gray')
        axes[0, i].set_title(f"1P-F{i}")
        axes[0, i].axis('off')
        
        # 2Pのフレームを表示
        axes[1, i].imshow(imgs[i+4], cmap='gray')
        axes[1, i].set_title(f"2P-F{i}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- クリティックモデル定義 ---
class CentralizedCritic(nn.Module):
    def __init__(self, input_channels=8): 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, joint_obs):
        # バッチ次元がない場合は追加 [8, 84, 84] -> [1, 8, 84, 84]
        if joint_obs.dim() == 3:
            joint_obs = joint_obs.unsqueeze(0)
        features = self.encoder(joint_obs)
        value = self.fc(features)
        return value

# --- 環境の準備 ---
def get_env():
    env = boxing_v2.parallel_env()
    env = ss.resize_v1(env, 84, 84)
    env = ss.color_reduction_v0(env, mode='full')
    env = ss.dtype_v0(env, "float32")
    env = ss.frame_stack_v1(env, 4)
    return ParallelPettingZooEnv(env)

# --- 実行確認 ---
env = get_env()
obs_dict, info = env.reset()

# 統合処理の実行
obs_1p, obs_2p, joint_s = preprocess_joint_obs(obs_dict)

print(f"Original Obs Shape: {obs_dict['first_0'].shape}") # (84, 84, 4)
print(f"Processed 1P Shape: {obs_1p.shape}")            # (4, 84, 84)
print(f"Joint State Shape: {joint_s.shape}")             # (8, 84, 84)

# 可視化して確認
visualize_check(joint_s)