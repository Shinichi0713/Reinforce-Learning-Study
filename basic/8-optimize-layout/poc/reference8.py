import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os

GRID_SIZE = 10
Number_of_Rectangles = 15

def generate_random_rects(min_n=2, max_n=5, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects

# 方策ネットワーク（Softmaxで確率分布を出力）
class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_nn.pth')
        self.__load_state_dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        logits = self.fc(x)
        return torch.softmax(logits, dim=1)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            print("load network parameters")
            self.load_state_dict(torch.load(self.path_nn))

    def save_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

def select_action(probs):
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def apply_action(state, action, rects):
    grid = state.copy()
    # rect_idx = action % len(rects)
    # pos_idx = action // len(rects)
    rect_idx = action % Number_of_Rectangles
    pos_idx = action // Number_of_Rectangles
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    if rect_idx >= len(rects) - 1:
        return grid, -1, False
    w, h = rects[rect_idx]
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -2, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    return grid, ratio_filled*10, True


def train():
    num_actions = GRID_SIZE * GRID_SIZE * Number_of_Rectangles
    policy_net = PolicyNet(num_actions)
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    for episode in range(200):
        # ランダムな長方形リストと配置回数を決定
        rects = generate_random_rects()
        max_steps = random.randint(5, 15)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        log_probs = []
        rewards = []
        total_reward = 0
        loss_total = 0.0
        for t in range(max_steps):
            state_tensor = torch.tensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            action, log_prob = select_action(probs)
            next_state, reward, success = apply_action(state, action, rects)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if not success:
                break
        # 報酬和を割引せずそのまま（単純化のため）
        G = sum(rewards)
        loss = -torch.stack(log_probs).sum() * G
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"episode {episode} total reward: {total_reward} loss: {loss_total / 10}")
            loss_total = 0.0
    policy_net.save_state_dict()


def eval():
    rects = generate_random_rects()
    num_actions = GRID_SIZE * GRID_SIZE * Number_of_Rectangles
    policy_net = PolicyNet(num_actions)
    
    # テスト（最後のエピソードのrectsとpolicy_netを利用）
    policy_net.eval()
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for i in range(5):
        state_tensor = torch.tensor(state).unsqueeze(0)
        probs = policy_net(state_tensor)
        action, log_prob = select_action(probs)
        next_state, reward, success = apply_action(state, action, rects)
        print(f"action: {action}, reward: {reward}")
        state = next_state

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('optimal arrangement')
    plt.show()


if __name__ == "__main__":
    train()
    eval()
