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
        return torch.softmax(x, dim=1)

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

def select_action(probs):
    # probs: (1, NUM_ACTIONS)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


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
policy_net = QNet()
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4)
BATCH = 32
MEMORY = []
epsilon = 0.85
for episode in range(200):
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    log_probs = []
    rewards = []
    total_reward = 0
    for t in range(10):
        state_tensor = torch.tensor(state).unsqueeze(0)
        probs = policy_net(state_tensor)
        action, log_prob = select_action(probs)
        next_state, reward, success = apply_action(state, action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state
        if not success:
            break
    # 報酬和を割引せずそのまま（単純化のため）
    G = sum(rewards)
    loss = -torch.stack(log_probs).sum() * G
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if episode % 50 == 0:
        print(f"episode {episode} total reward: {total_reward}")
    policy_net.save_to_state_dict()


# 最終配置例の可視化
import matplotlib.pyplot as plt
state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
for i in range(3):
    state_tensor = torch.tensor(state).unsqueeze(0)  # (1,1,H,W)
    qvals = policy_net(state_tensor)
    action = get_action(qvals, epsilon=0)
    print(f"action: {action}")
    next_state, reward, success = apply_action(state, action)
    state = next_state

plt.imshow(state[0])
plt.title('optimal arrangement')
plt.show()
