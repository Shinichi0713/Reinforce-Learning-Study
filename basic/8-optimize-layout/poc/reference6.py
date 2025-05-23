import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# パラメータ
GRID_SIZE = 10
RECTS = [(2,2), (3,1), (1,3)]
NUM_ACTIONS = GRID_SIZE * GRID_SIZE * len(RECTS)

# Qネットワーク
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128), nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.fc(x)
        return x

def get_action(qvals, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS-1)
    else:
        return torch.argmax(qvals).item()

def apply_action(state, action):
    grid = state.copy()
    rect_idx = action % len(RECTS)
    pos_idx = action // len(RECTS)
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    w, h = RECTS[rect_idx]
    # 範囲外なら失敗
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    # 重なりチェック
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -1, False
    # 配置
    grid[0, y:y+h, x:x+w] = 1
    return grid, 1, True

# 簡易DQNループ
qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
BATCH = 32
MEMORY = []

for episode in range(200):
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    total_reward = 0
    for t in range(10):
        state_tensor = torch.tensor(state).unsqueeze(0)  # (1,1,H,W)
        qvals = qnet(state_tensor)
        action = get_action(qvals, epsilon=0.2)
        next_state, reward, success = apply_action(state, action)
        total_reward += reward
        MEMORY.append((state, action, reward, next_state))
        state = next_state
        if not success:
            break
        # 簡易学習
        if len(MEMORY) >= BATCH:
            batch = random.sample(MEMORY, BATCH)
            batch_s = torch.tensor([b[0] for b in batch]).to(qnet.device)
            batch_a = torch.tensor([b[1] for b in batch]).to(qnet.device)
            batch_r = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(qnet.device)
            batch_ns = torch.tensor([b[3] for b in batch]).to(qnet.device)
            q_pred = qnet(batch_s).gather(1, batch_a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = qnet(batch_ns).max(1)[0]
            loss = ((q_pred - (batch_r + 0.9 * q_next))**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if episode % 50 == 0:
        print(f"episode {episode} total reward: {total_reward}")

# 最終配置例の可視化
import matplotlib.pyplot as plt
plt.imshow(state[0])
plt.title('optimal arrangement')
plt.show()
