
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque, namedtuple
GRID_SIZE = 10
MAX_RECTS = 5  # 最大箱数


class PolicyNet(nn.Module):
    def __init__(self, num_actions, max_rects=5):
        super().__init__()
        # 画像用CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 32 * GRID_SIZE * GRID_SIZE), nn.ReLU()
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
        return logits

    def save_state_dict(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            print("load network parameter")
            self.load_state_dict(torch.load(self.path_nn))


class ActorNet(nn.Module):
    def __init__(self, num_actions, max_rects=5):
        super().__init__()
        self.policy_net = PolicyNet(num_actions, max_rects)
    
    def forward(self, grid, rects_info):
        # 行動確率分布
        return torch.softmax(self.policy_net(grid, rects_info), dim=1)  # softmax済み


class CriticNet(nn.Module):
    def __init__(self, num_actions, max_rects=5):
        super().__init__()
        self.q_net = PolicyNet(num_actions, max_rects)
    
    def forward(self, grid, rects_info):
        return self.q_net(grid, rects_info)  # Q値（softmaxしないこと）


def generate_random_rects(min_n=2, max_n=MAX_RECTS, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects


# AIエージェントの予測確率に基づき対数確率を計算する
def select_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.randrange(q_values.shape[1])
    else:
        return q_values.argmax(1).item()


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
    # サイズチェック
    if w == 0 or h == 0:
        return grid, -2, False
    # 重なりチェック
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -3, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    return grid, ratio_filled*18, True


def update_soft_target(target_net, source_net, tau=0.03):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    return target_net

Transition = namedtuple('Transition', ('state', 'rects', 'action', 'reward', 'next_state', 'next_rects', 'done'))

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


def train_sac():
    num_episodes = 1000
    num_actions = GRID_SIZE * GRID_SIZE * MAX_RECTS
    actor = ActorNet(num_actions)
    critic1 = CriticNet(num_actions)
    critic2 = CriticNet(num_actions)
    target_critic1 = CriticNet(num_actions)
    target_critic2 = CriticNet(num_actions)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=1e-3)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=1e-3)
    alpha = 0.2  # 固定値（自動調整も可）
    gamma = 0.99
    tau = 0.03
    buffer = ReplayBuffer(10000)
    BATCH_SIZE = 64

    for episode in range(num_episodes):
        rects = generate_random_rects()
        num_rects = len(rects)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        rects_input = np.concatenate([rects_info, [num_rects]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)
        total_reward = 0
        max_steps = random.randint(5, 9)
        for t in range(max_steps):
            state_tensor = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = actor(state_tensor, rects_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

            next_state, reward, success = apply_action(state, action, rects_info.tolist())
            done = not success or t == max_steps - 1
            buffer.push(state, rects_input, action, reward, next_state, rects_input, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                batch_state = torch.tensor(np.stack(transitions.state)).float().to(actor.policy_net.device)
                batch_rects = torch.tensor(np.stack(transitions.rects)).float().to(actor.policy_net.device)
                batch_action = torch.tensor(transitions.action).unsqueeze(1).to(actor.policy_net.device)
                batch_reward = torch.tensor(transitions.reward).float().to(actor.policy_net.device)
                batch_next_state = torch.tensor(np.stack(transitions.next_state)).float().to(actor.policy_net.device)
                batch_next_rects = torch.tensor(np.stack(transitions.next_rects)).float().to(actor.policy_net.device)
                batch_done = torch.tensor(transitions.done).float().to(actor.policy_net.device)

                # --- Criticの更新 ---
                with torch.no_grad():
                    next_action_probs = actor(batch_next_state, batch_next_rects)
                    next_log_probs = torch.log(next_action_probs + 1e-10)
                    target_q1 = target_critic1(batch_next_state, batch_next_rects)
                    target_q2 = target_critic2(batch_next_state, batch_next_rects)
                    target_min_q = torch.min(target_q1, target_q2)
                    # V値 = Σ_a π(a|s) [ Q(s,a) - α log π(a|s) ]
                    next_v = (next_action_probs * (target_min_q - alpha * next_log_probs)).sum(dim=1)
                    target_q = batch_reward + gamma * (1 - batch_done) * next_v

                current_q1 = critic1(batch_state, batch_rects).gather(1, batch_action).squeeze(1)
                current_q2 = critic2(batch_state, batch_rects).gather(1, batch_action).squeeze(1)
                critic1_loss = nn.MSELoss()(current_q1, target_q)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # --- Actorの更新 ---
                action_probs = actor(batch_state, batch_rects)
                log_probs = torch.log(action_probs + 1e-10)
                q1 = critic1(batch_state, batch_rects)
                q2 = critic2(batch_state, batch_rects)
                min_q = torch.min(q1, q2)
                actor_loss = (action_probs * (alpha * log_probs - min_q)).sum(dim=1).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- ターゲットネットのソフト更新 ---
                target_critic1 = update_soft_target(target_critic1, critic1, tau)
                target_critic2 = update_soft_target(target_critic2, critic2, tau)

        if episode % 10 == 0:
            print(f"episode {episode} total reward: {total_reward:.2f}")
    print("Training finished.")


def eval():
    rects = generate_random_rects()
    num_rects = len(rects)
    num_actions = GRID_SIZE * GRID_SIZE * MAX_RECTS
    policy_net = PolicyNet(num_actions)
    policy_net.eval()
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    action_last = 0
    reward_last = 0
    rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
    for i, (w, h) in enumerate(rects):
        rects_info[i*2:i*2+2] = [w, h]
    rects_input = np.concatenate([rects_info, [num_rects]]).astype(np.float32)
    rects_tensor = torch.tensor(rects_input).unsqueeze(0)
    print(f"Evaluating with {num_rects} rectangles: {rects}")
    for i in range(7):
        state_tensor = torch.tensor(state).unsqueeze(0)  # (1, 1, H, W)
        q_values = policy_net(state_tensor, rects_tensor)
        action = select_action(q_values, 0.0)  # 評価時はε=0
        next_state, reward, success = apply_action(state, action, rects_info.tolist())
        print(f"action: {action}, reward: {reward}")
        state = next_state

        # action_last = 0
        # reward_last = 0
        # rects_tensor = torch.tensor(np.concatenate([rects_info, [num_rects, action_last/100, reward_last]]).astype(np.float32)).unsqueeze(0)

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('Final arrangement')
    plt.show()


if __name__ == "__main__":
    train_sac()
    eval()

