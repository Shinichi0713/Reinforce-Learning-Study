import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import time

# ---------- ハイパーパラメータ ----------
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_AGENTS = 2           # 環境と合わせる（env.num_agents）
N_ACTIONS = 5          # env.action_space (0..4)
OBS_DIM = None         # 後で env.reset() から決定
STATE_DIM = None       # 後で env._get_state() の長さから決定

EPISODES = 800
MAX_STEPS = 40
BATCH_SIZE = 32
BUFFER_CAP = 5000
GAMMA = 0.99
LR = 5e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 200   # 学習ステップごとのターゲット更新
TRAIN_START = 1000         # バッファがこれだけ貯まったら学習開始
TRAIN_FREQ = 1             # 何ステップごとに学習するか

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Replay Buffer ----------
Transition = namedtuple('Transition', ['obs', 'state', 'actions', 'rewards', 'next_obs', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, cap=BUFFER_CAP):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        return batch
    def __len__(self): return len(self.buf)

# ---------- 各エージェントのQネットワーク ----------
class AgentQNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)  # (batch, n_actions)
    
    