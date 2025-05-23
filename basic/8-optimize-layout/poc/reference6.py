import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os

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
        self.path_nn = os.path.join(os.path.dirname(__file__), "qnet.pth")
        self.__load_state_dict()

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.fc(x)
        return x
    
    def save_to_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            print("load nn parameter")
            self.load_state_dict(torch.load(self.path_nn))

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
        return grid, -2, False
    # 配置
    grid[0, y:y+h, x:x+w] = 1
    return grid, 1, True

# 簡易DQNループ
qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
BATCH = 32
MEMORY = []
epsilon = 0.85
for episode in range(400):
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    total_reward = 0
    loss_total = 0.0
    for t in range(20):
        state_tensor = torch.tensor(state).unsqueeze(0)  # (1,1,H,W)
        qvals = qnet(state_tensor)
        action = get_action(qvals, epsilon=epsilon)
        epsilon *= 0.99
        next_state, reward, success = apply_action(state, action)
        total_reward += reward
        MEMORY.append((state, action, reward, next_state))
        state = next_state
        if not success:
            break
        # 簡易学習
        if len(MEMORY) >= BATCH:
            batch = random.sample(MEMORY, BATCH)
            batch_s = torch.tensor(np.array([b[0] for b in batch])).to(qnet.device)
            batch_a = torch.tensor(np.array([b[1] for b in batch])).to(qnet.device)
            batch_r = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(qnet.device)
            batch_ns = torch.tensor(np.array([b[3] for b in batch])).to(qnet.device)
            q_pred = qnet(batch_s).gather(1, batch_a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = qnet(batch_ns).max(1)[0]
            loss = (torch.abs(q_pred - (batch_r + 0.9 * q_next))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

    if episode % 10 == 0:
        print(f"episode {episode} loss: {loss_total / 10}")
        # print(f"episode {episode} total reward: {total_reward}")
        loss_total = 0.0
    qnet.save_to_state_dict()


# 最終配置例の可視化
import matplotlib.pyplot as plt
for rect in RECTS:
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=True, color='red', alpha=0.5))
plt.imshow(state[0])
plt.title('optimal arrangement')
plt.show()
