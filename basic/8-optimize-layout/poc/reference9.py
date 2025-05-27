
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os

GRID_SIZE = 10
MAX_RECTS = 5  # 最大箱数


class PolicyNet(nn.Module):
    def __init__(self, num_actions, max_rects=5):
        super().__init__()
        # 画像用CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # 箱情報用MLP
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        # 結合後のFC
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 32, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        # ...（省略: パラメータ保存/ロード等）
        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_nn.pth')

    def forward(self, grid, rects_info):
        # grid: (B, 1, H, W)
        # rects_info: (B, max_rects * 2 + 1)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        logits = self.fc(x)
        return torch.softmax(logits, dim=1)


def generate_random_rects(min_n=2, max_n=MAX_RECTS, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects


def select_action(probs):
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def apply_action(state, action, rects):
    grid = state.copy()
    num_rects = len(rects)
    num_actions = GRID_SIZE * GRID_SIZE * num_rects
    rect_idx = action % num_rects
    pos_idx = action // num_rects
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    w, h = rects[rect_idx]
    # 範囲外チェック
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    # 重なりチェック
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -2, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    return grid, ratio_filled*10, True


def train():
    num_episodes = 200
    for episode in range(num_episodes):
        # 対象となる箱を生成
        rects = generate_random_rects()
        num_rects = len(rects)
        # この段階でアクションを生成
        num_actions = GRID_SIZE * GRID_SIZE * num_rects
        # ここで定義がいただけない
        policy_net = PolicyNet(num_actions)
        optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

        # 状態初期化
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        log_probs = []
        rewards = []
        total_reward = 0

        # 箱情報ベクトル化
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        # 箱数を追加
        rects_input = np.concatenate([rects_info, [num_rects]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)  # (1, MAX_RECTS*2+1)
        # 画像状態と箱情報をネットワークに通す
        max_steps = random.randint(5, 15)
        for t in range(max_steps):
            state_tensor = torch.tensor(state).unsqueeze(0)  # (1, 1, H, W)
            probs = policy_net(state_tensor, rects_tensor)
            action, log_prob = select_action(probs)
            next_state, reward, success = apply_action(state, action, rects)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if not success:
                break

        # リターン計算（単純合計）
        G = sum(rewards)
        loss = -torch.stack(log_probs).sum() * G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"episode {episode} total reward: {total_reward:.2f} loss: {loss.item():.3f}")

    print("Training finished.")


def eval():
    rects = generate_random_rects()
    num_rects = len(rects)
    num_actions = GRID_SIZE * GRID_SIZE * num_rects
    policy_net = PolicyNet(num_actions)
    policy_net.eval()
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
    for i, (w, h) in enumerate(rects):
        rects_info[i*2:i*2+2] = [w, h]
    rects_input = np.concatenate([rects_info, [num_rects]]).astype(np.float32)
    rects_tensor = torch.tensor(rects_input).unsqueeze(0)

    for i in range(5):
        state_tensor = torch.tensor(state).unsqueeze(0)
        probs = policy_net(state_tensor, rects_tensor)
        action, _ = select_action(probs)
        next_state, reward, success = apply_action(state, action, rects)
        print(f"action: {action}, reward: {reward}")
        state = next_state

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('Final arrangement')
    plt.show()


if __name__ == "__main__":
    train()
    eval()

