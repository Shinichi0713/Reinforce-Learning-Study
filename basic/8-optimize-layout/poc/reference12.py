import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque, namedtuple

GRID_SIZE = 10
MAX_RECTS = 5  # 最大箱数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ネットワーク定義 ---
class BaseNet(nn.Module):
    def __init__(self, num_actions, max_rects=5):
        super().__init__()
        # 画像用CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # 箱情報用MLP
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 3, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 64), nn.ReLU()
        )
        self.num_actions = num_actions
        self.fc_in_dim = 64 * GRID_SIZE * GRID_SIZE + 64

class ActorNet(BaseNet):
    def __init__(self, num_actions, max_rects=5):
        super().__init__(num_actions, max_rects)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_dim, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, num_actions)
        )
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, "model_actor.pth")
        self.load_model()
        self.to(DEVICE)

    def forward(self, grid, rects_info):
        grid = grid.to(DEVICE)
        rects_info = rects_info.to(DEVICE)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)  # (B, num_actions)
        return probs
    
    def save_model(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_model)
        self.to(DEVICE)

    def load_model(self):
        if os.path.exists(self.path_model):
            print(f"Loading model from {self.path_model}")
            self.load_state_dict(torch.load(self.path_model, map_location=DEVICE))
        else:
            print(f"Model file {self.path_model} does not exist. Skipping load.")

class CriticNet(BaseNet):
    def __init__(self, num_actions, max_rects=5, name ="critic"):
        super().__init__(num_actions, max_rects)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_dim, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, num_actions)
        )
        
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, f"model_{name}.pth")
        self.load_model()
        self.to(DEVICE)

    def forward(self, grid, rects_info):
        grid = grid.to(DEVICE)
        rects_info = rects_info.to(DEVICE)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        q_values = self.fc(x)
        return q_values
    
    def save_model(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_model)
        self.to(DEVICE)

    def load_model(self):
        if os.path.exists(self.path_model):
            self.load_state_dict(torch.load(self.path_model, map_location=DEVICE))
        else:
            print(f"Model file {self.path_model} does not exist. Skipping load.")

# --- 環境・バッファ ---
def generate_random_rects(min_n=2, max_n=MAX_RECTS, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects

def select_action(probs, epsilon):
    if random.random() < epsilon:
        return random.randrange(probs.shape[1])
    else:
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

def apply_action(state, action, rects):
    grid = state.copy()
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

# --- メイン学習ループ ---
def train():
    num_episodes = 10000
    num_actions = GRID_SIZE * GRID_SIZE * MAX_RECTS
    actor = ActorNet(num_actions, MAX_RECTS).to(DEVICE)
    critic1 = CriticNet(num_actions, MAX_RECTS, name="critic1").to(DEVICE)
    critic2 = CriticNet(num_actions, MAX_RECTS, name="critic2").to(DEVICE)
    target_critic1 = CriticNet(num_actions, MAX_RECTS, name="target_critic1").to(DEVICE)
    target_critic2 = CriticNet(num_actions, MAX_RECTS, name="target_critic2").to(DEVICE)
    # target_critic1.load_state_dict(critic1.state_dict())
    # target_critic2.load_state_dict(critic2.state_dict())

    actor_optimizer = optim.AdamW(actor.parameters(), lr=1e-4)
    critic1_optimizer = optim.AdamW(critic1.parameters(), lr=1e-3)
    critic2_optimizer = optim.AdamW(critic2.parameters(), lr=1e-3)

    buffer = ReplayBuffer(10000)
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON = 0.8
    ALPHA = 0.2  # エントロピー温度

    for episode in range(num_episodes):
        rects = generate_random_rects()
        num_rects = len(rects)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        action_last = 0
        reward_last = 0
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        rects_input = np.concatenate([rects_info, [num_rects, action_last/100, reward_last]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)
        
        total_reward = 0
        max_steps = random.randint(5, 9)
        
        for t in range(max_steps):
            state_tensor = torch.tensor(state).unsqueeze(0)  # (1, 1, H, W)
            probs = actor(state_tensor, rects_tensor)
            action = select_action(probs, EPSILON)
            EPSILON = max(EPSILON * 0.995, 0.05)  # ε-greedyの減衰
            next_state, reward, success = apply_action(state, action, rects_info.tolist())
            next_state_tensor = torch.tensor(next_state).unsqueeze(0)
            done = not success or t == max_steps - 1
            buffer.push(state, rects_input, action, reward, next_state, rects_input, done)
            state = next_state
            total_reward += reward
            count_rects = ((rects_info != 0).sum().item()) // 2
            if count_rects == 0:
                break
            # --- 学習 ---
            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                batch_state = torch.tensor(np.stack(transitions.state)).float().to(DEVICE)
                batch_rects = torch.tensor(np.stack(transitions.rects)).float().to(DEVICE)
                batch_action = torch.tensor(transitions.action).unsqueeze(1).to(DEVICE)
                batch_reward = torch.tensor(transitions.reward).float().to(DEVICE)
                batch_next_state = torch.tensor(np.stack(transitions.next_state)).float().to(DEVICE)
                batch_next_rects = torch.tensor(np.stack(transitions.next_rects)).float().to(DEVICE)
                batch_done = torch.tensor(transitions.done).bool().to(DEVICE)

                # --- Critic損失 ---
                with torch.no_grad():
                    next_probs = actor(batch_next_state, batch_next_rects)  # (B, num_actions)
                    next_log_probs = torch.log(next_probs + 1e-8)
                    next_q1 = target_critic1(batch_next_state, batch_next_rects)
                    next_q2 = target_critic2(batch_next_state, batch_next_rects)
                    next_min_q = torch.min(next_q1, next_q2)
                    next_v = (next_probs * (next_min_q - ALPHA * next_log_probs)).sum(dim=1)
                    target_q = batch_reward + GAMMA * next_v * (~batch_done)

                q1 = critic1(batch_state, batch_rects)
                q2 = critic2(batch_state, batch_rects)
                q1_pred = q1.gather(1, batch_action).squeeze(1)
                q2_pred = q2.gather(1, batch_action).squeeze(1)
                critic1_loss = nn.MSELoss()(q1_pred, target_q)
                critic2_loss = nn.MSELoss()(q2_pred, target_q)

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # --- Actor損失 ---
                probs = actor(batch_state, batch_rects)
                log_probs = torch.log(probs + 1e-8)
                q1_detached = critic1(batch_state, batch_rects).detach()
                q2_detached = critic2(batch_state, batch_rects).detach()
                min_q = torch.min(q1_detached, q2_detached)
                actor_loss = (probs * (ALPHA * log_probs - min_q)).sum(dim=1).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- ターゲットネットのソフト更新 ---
                update_soft_target(target_critic1, critic1, tau=0.1)
                update_soft_target(target_critic2, critic2, tau=0.1)

        if episode % 10 == 0:
            print(f"episode {episode} total reward: {total_reward:.2f}")
        if episode % 100 == 0:
            actor.save_model()
            critic1.save_model()
            critic2.save_model()
            target_critic1.save_model()
            target_critic2.save_model()
    print("Training finished.")

if __name__ == "__main__":
    train()
