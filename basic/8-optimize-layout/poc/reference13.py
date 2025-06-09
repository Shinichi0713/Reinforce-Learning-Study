## ddqnで箱配置すると、箱が消えるようにする
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque, namedtuple
GRID_SIZE = 10
MAX_RECTS = 5  # 最大箱数


# 方策ネットワーク
import torch
import torch.nn as nn
import os


class PolicyNet(nn.Module):
    def __init__(self, size_grid, max_rects=5):
        super().__init__()
        # 画像用CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # 箱情報用MLP
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 3, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 64), nn.ReLU()
        )
        # 配置予定の箱を選択
        self.select_box = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, max_rects)
        )
        # 配置位置を決定
        self.place_box = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, size_grid)
        )
        # ...（省略: パラメータ保存/ロード等）
        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_nn.pth')
        self.__load_state_dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, grid, rects_info):
        # grid: (B, 1, H, W)
        # rects_info: (B, max_rects * 2 + 3)
        grid = grid.to(self.device)
        rects_info = rects_info.to(self.device)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        
        logits_box = self.select_box(x)
        logits_place = self.place_box(x)

        return logits_box, logits_place

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
def select_action(logits_box, logits_place, epsilon):
    if random.random() < epsilon:
        return random.randrange(logits_box.shape[1]), random.randrange(logits_place.shape[1])
    else:
        return logits_box.argmax(1).item(), logits_place.argmax(1).item()


def apply_action(state, rects, index_box, index_place):
    grid = state.copy()
    # num_rects = len(rects)
    # num_actions = GRID_SIZE * GRID_SIZE * num_rects
    rect_idx = index_box
    pos_idx = index_place
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    w, h = int(rects[rect_idx*2]), int(rects[rect_idx*2+1])
    # 範囲外チェック
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    # サイズチェック
    if w == 0 or h == 0:
        return grid, -2, False
    # 重なりチェック
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -3, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    # 箱の中身を空白か
    rects[rect_idx*2:rect_idx*2+2] = [0, 0]  # 箱を空にする
    # 箱の中身が全て0か
    if all(r == 0 for r in rects):
        reward_complete = 10
    else:
        reward_complete = 0
    return grid, ratio_filled*18 + reward_complete, True


def update_soft_target(target_net, source_net, tau=0.03):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


Transition = namedtuple('Transition', ('state', 'rects', 'index_box', 'index_place', 'reward', 'next_state', 'next_rects', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)


def train():
    num_episodes = 80000
    size_grid = GRID_SIZE * GRID_SIZE
    q_net = PolicyNet(size_grid, MAX_RECTS)
    q_net.train()
    target_net = PolicyNet(size_grid, MAX_RECTS)
    target_net.eval()
    optimizer = optim.AdamW(q_net.parameters(), lr=2e-4)
    # 学習率は減衰させる
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.96)

    buffer = ReplayBuffer(10000)
    BATCH_SIZE = 32
    GAMMA = 0.9         # 割引率: デフォルト 0.99
    EPSILON = 0.95
    TARGET_UPDATE = 10

    for episode in range(num_episodes):
        rects = generate_random_rects()
        num_rects = len(rects)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        rects_input = np.concatenate([rects_info, [num_rects, 0.0, 0.0]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)
        total_reward = 0
        # max_steps = random.randint(5, 9)
        for i in range(num_rects + 3):
            state_tensor = torch.tensor(state).unsqueeze(0)  # (1, 1, H, W)
            logits_box, logits_place = q_net(state_tensor, rects_tensor)
            index_box, index_place = select_action(logits_box, logits_place, EPSILON)

            next_state, reward, success = apply_action(state, rects_info, index_box, index_place)
            next_state_tensor = torch.tensor(next_state).unsqueeze(0)
            count_rects = ((rects_info != 0).sum().item()) // 2
            rects_input_next = np.concatenate([rects_info.copy(), [count_rects, index_box, 0.0]]).astype(np.float32)
            rects_tensor = torch.tensor(rects_input_next).unsqueeze(0)

            done = not success or i == num_rects - 1
            buffer.push(state, rects_input, index_box, index_place, reward, next_state, rects_input_next, done)
            state = next_state
            total_reward += reward
            # 学習
            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                batch_state = torch.tensor(np.stack(transitions.state)).float().to(q_net.device)
                batch_rects = torch.tensor(np.stack(transitions.rects)).float().to(q_net.device)
                batch_index_box = torch.tensor(transitions.index_box).unsqueeze(1).to(q_net.device)
                batch_index_place = torch.tensor(transitions.index_place).unsqueeze(1).to(q_net.device)
                batch_reward = torch.tensor(transitions.reward).float().to(q_net.device)
                batch_next_state = torch.tensor(np.stack(transitions.next_state)).float().to(q_net.device)
                batch_next_rects = torch.tensor(np.stack(transitions.next_rects)).float().to(q_net.device)
                batch_done = torch.tensor(transitions.done).bool().to(q_net.device)

                # Double DQNのターゲット計算
                with torch.no_grad():
                    logits_box, logits_place = q_net(batch_next_state, batch_next_rects)
                    next_choice_box = logits_box.argmax(1, keepdim=True)
                    next_choice_place = logits_place.argmax(1, keepdim=True)
                    next_q_target_box, next_q_target_place = target_net(batch_next_state, batch_next_rects)
                    next_q_target_box = next_q_target_box.gather(1, next_choice_box).squeeze(1)
                    next_q_target_place = next_q_target_place.gather(1, next_choice_place).squeeze(1)
                    expected_box = batch_reward + GAMMA * next_q_target_box
                    expected_place = batch_reward + GAMMA * next_q_target_place

                q_pred_box, q_pred_place = q_net(batch_state, batch_rects)
                q_pred_box = q_pred_box.gather(1, batch_index_box).squeeze(1)
                q_pred_place = q_pred_place.gather(1, batch_index_place).squeeze(1)
                loss_box = nn.MSELoss()(q_pred_box, expected_box)
                loss_place = nn.MSELoss()(q_pred_place, expected_place)
                loss = loss_box + loss_place
                # print(f"episode {episode}, step {t}, loss: {loss.item():.4f}, reward: {reward:.2f}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                scheduler.step()  # 学習率の更新

                # Soft targetネットワークの更新
                update_soft_target(target_net, q_net, tau=0.1)

            # 残り箱がないなら終了
            if count_rects == 0:
                break

        if episode % 10 == 0:
            EPSILON *= 0.999  # ε-greedyの減衰
            q_net.save_state_dict()
            print(f"episode {episode} total reward: {total_reward:.2f} epsilon: {EPSILON:.2f}")
    print("Training finished.")


def eval():
    rects = generate_random_rects()
    num_rects = len(rects)
    size_grid = GRID_SIZE * GRID_SIZE
    q_net = PolicyNet(size_grid, MAX_RECTS)
    q_net.eval()
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    action_last = 0
    reward_last = 0
    rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
    for i, (w, h) in enumerate(rects):
        rects_info[i*2:i*2+2] = [w, h]
    count_rects = ((rects_info != 0).sum().item()) // 2
    rects_input = np.concatenate([rects_info, [count_rects, action_last/100, reward_last]]).astype(np.float32)
    rects_tensor = torch.tensor(rects_input).unsqueeze(0)
    print(f"Evaluating with {num_rects} rectangles: {rects}")
    for i in range(num_rects):
        state_tensor = torch.tensor(state).unsqueeze(0)  # (1, 1, H, W)
        logits_box, logits_place = q_net(state_tensor, rects_tensor)
        index_box, index_place = select_action(logits_box, logits_place, 0.0)
        # apply_action(state, rects, index_box, index_place)
        next_state, reward, success = apply_action(state, rects_info, index_box, index_place)
        rects_input = np.concatenate([rects_info, [num_rects-i-1, action_last/100, reward_last]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)

        print(f"choice_box: {index_box}, arrange_box: {index_place}, reward: {reward}")
        state = next_state

        # action_last = action
        # reward_last = reward
        # rects_tensor = torch.tensor(np.concatenate([rects_info, [num_rects, action_last/100, reward_last]]).astype(np.float32)).unsqueeze(0)

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('Final arrangement')
    plt.show()


if __name__ == "__main__":
    train()
    eval()

