
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
            nn.Linear(max_rects * 2 + 1, 64 * 2), nn.ReLU(),
            nn.Linear(64 * 2, 64 * 2), nn.ReLU(),
            nn.Linear(64 * 2, 64), nn.ReLU()
        )
        # 結合後のFC
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        # ...（省略: パラメータ保存/ロード等）
        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_nn.pth')
        self.__load_state_dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, grid, rects_info):
        # grid: (B, 1, H, W)
        # rects_info: (B, max_rects * 2 + 1)
        grid = grid.to(self.device)
        rects_info = rects_info.to(self.device)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        logits = self.fc(x)
        return torch.softmax(logits, dim=1)

    def save_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            print("load network parameter")
            self.load_state_dict(torch.load(self.path_nn))


def generate_random_rects(min_n=2, max_n=MAX_RECTS, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects


# AIエージェントの予測確率に基づき対数確率を計算する
def select_action(probs):
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    # print(torch.max(probs))
    return action.item(), m.log_prob(action)

def apply_action(state, action, rects):
    grid = state.copy()
    # num_rects = len(rects)
    # num_actions = GRID_SIZE * GRID_SIZE * num_rects
    rect_idx = action % MAX_RECTS
    pos_idx = action // MAX_RECTS
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    w, h = int(rects[rect_idx*2]), int(rects[rect_idx*2+1])
    # 範囲外チェック
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    if w == 0 or h == 0:
        return grid, -2, False
    # 重なりチェック
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -3, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    return grid, ratio_filled*14, True


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

def train():
    num_episodes = 1000
    batch_size = 32
    replay_capacity = 10000
    num_actions = GRID_SIZE * GRID_SIZE * MAX_RECTS
    policy_net = PolicyNet(num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(replay_capacity)

    for episode in range(num_episodes):
        rects = generate_random_rects()
        num_rects = len(rects)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # 箱情報ベクトル化
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        rects_input = np.concatenate([rects_info, [num_rects]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)
        max_steps = random.randint(5, 8)
        total_reward = 0
        episode_transitions = []
        for t in range(max_steps):
            state_tensor = torch.tensor(state).unsqueeze(0)
            probs = policy_net(state_tensor, rects_tensor)
            action, log_prob = select_action(probs)
            next_state, reward, success = apply_action(state, action, rects_info.tolist())
            total_reward += reward
            done = not success or t == max_steps - 1
            # 状態、行動、log_prob、報酬、次状態、done, rects_info
            replay_buffer.push(state.copy(), action, log_prob, reward, next_state.copy(), done, rects_input.copy())
            state = next_state
            if done:
                break

        # バッファが十分溜まったらランダムサンプルで学習
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, log_probs, rewards, next_states, dones, rects_inputs = batch

            # バッチデータをテンソル化
            state_batch = torch.tensor(np.array(states)).float()
            rects_batch = torch.tensor(np.array(rects_inputs)).float()
            log_prob_batch = torch.stack(log_probs)
            reward_batch = torch.tensor(rewards).float()
            # 割引累積報酬計算（ここでは単純な報酬合計でも可）
            returns = []
            G = 0
            gamma = 0.9
            for r, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    G = 0
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns).float()

            # 方策勾配損失
            loss = -torch.sum(log_prob_batch * returns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"episode {episode} total reward: {total_reward:.2f} buffer: {len(replay_buffer)}")
            policy_net.save_state_dict()

    print("Training finished.")


def eval():
    rects = generate_random_rects()
    num_rects = len(rects)
    num_actions = GRID_SIZE * GRID_SIZE * MAX_RECTS
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
        next_state, reward, success = apply_action(state, action, rects_info.tolist())
        print(f"action: {action}, reward: {reward}")
        state = next_state

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('Final arrangement')
    plt.show()


if __name__ == "__main__":
    train()
    eval()

