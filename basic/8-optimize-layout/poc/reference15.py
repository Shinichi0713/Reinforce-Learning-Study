import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque, namedtuple

GRID_SIZE = 10
MAX_RECTS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ネットワーク定義 ---
class ActorNet(nn.Module):
    def __init__(self, size_grid, max_rects=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 3, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 64), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.GELU(),
        )
        self.box_head = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, max_rects)
        )
        self.place_head = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 2, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, size_grid)
        )

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, "model_actor.pth")

    def forward(self, grid, rects_info):
        grid = grid.to(DEVICE)
        rects_info = rects_info.to(DEVICE)
        batch_size = grid.shape[0]
        contents_rect = rects_info[:, :10]
        # 偶数番目（2i番目）だけ取り出す
        even_idx_values = contents_rect[:, ::2]
        # 0以外なら1、0なら0
        mask = (even_idx_values != 0).long()
        # grid特徴量
        grid_feat = self.conv(grid)  # [B, 32*GRID_SIZE*GRID_SIZE]

        # rects特徴量
        rect_feat = self.rect_encoder(rects_info)  # [B, 64]

        # grid+rectsでボックス選択
        x = torch.cat([grid_feat, rect_feat], dim=1)
        x = self.fc(x)  # [B, 512]
        box_logits = self.box_head(x)  # [B, max_rects]
        box_logits = torch.where(mask.bool(), box_logits, torch.tensor(float('-inf')))
        box_probs = torch.softmax(box_logits, dim=1)
        index_box = torch.argmax(box_logits, dim=1)  # [B]
        # バッチごとに該当ボックスサイズを抽出
        # rects_info: [B, max_rects*2 + 3] → boxごとに(x, y)が並ぶと仮定
        # 例: [x0, y0, x1, y1, ...] なので、index_box*2, index_box*2+1でx, y
        box_size = []
        for b in range(batch_size):
            idx = index_box[b]
            x_val = rects_info[b, idx*2]
            y_val = rects_info[b, idx*2+1]
            box_size.append(torch.stack([x_val, y_val]))
        box_size = torch.stack(box_size, dim=0)  # [B, 2]

        # grid特徴量 + box_sizeで配置位置推定
        place_input = torch.cat([grid_feat, box_size], dim=1)
        place_logits = self.place_head(place_input)
        place_probs = torch.softmax(place_logits, dim=1)

        return box_probs, place_probs

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

class CriticNet(nn.Module):
    def __init__(self, size_grid, max_rects=5, name ="critic"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 3, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 64), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 512), nn.ReLU()
        )
        self.box_head = nn.Linear(512, max_rects)
        self.place_head = nn.Linear(512, size_grid)

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, f"model_{name}.pth")

    def forward(self, grid, rects_info):
        grid = grid.to(DEVICE)
        rects_info = rects_info.to(DEVICE)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        x = self.fc(x)
        box_q = self.box_head(x)
        place_q = self.place_head(x)
        return box_q, place_q

    def save_model(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_model)
        self.to(DEVICE)

    def load_model(self):
        if os.path.exists(self.path_model):
            self.load_state_dict(torch.load(self.path_model, map_location=DEVICE))
        else:
            print(f"Model file {self.path_model} does not exist. Skipping load.")

# --- その他の関数・バッファ ---
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

def generate_random_rects(min_n=2, max_n=MAX_RECTS, min_size=1, max_size=4):
    n = random.randint(min_n, max_n)
    rects = []
    for _ in range(n):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        rects.append((w, h))
    return rects

def select_action(box_probs, place_probs):
    box_dist = torch.distributions.Categorical(box_probs)
    place_dist = torch.distributions.Categorical(place_probs)
    box_action = box_dist.sample().item()
    place_action = place_dist.sample().item()
    return box_action, place_action

def apply_action(state, rects, index_box, index_place):
    grid = state.copy()
    rect_idx = index_box
    pos_idx = index_place
    x, y = pos_idx % GRID_SIZE, pos_idx // GRID_SIZE
    w, h = int(rects[rect_idx*2]), int(rects[rect_idx*2+1])
    if x + w > GRID_SIZE or y + h > GRID_SIZE:
        return grid, -1, False
    if w == 0 or h == 0:
        return grid, -2, False
    if np.any(grid[0, y:y+h, x:x+w] == 1):
        return grid, -3, False
    grid[0, y:y+h, x:x+w] = 1
    ratio_filled = np.sum(grid[0] == 1) / (GRID_SIZE * GRID_SIZE)
    rects[rect_idx*2:rect_idx*2+2] = [0, 0]
    if all(r == 0 for r in rects):
        reward_complete = 10
    else:
        reward_complete = 0
    return grid, ratio_filled*18 + reward_complete, True

# --- SAC学習 ---
def train():
    num_episodes = 20000
    size_grid = GRID_SIZE * GRID_SIZE
    actor = ActorNet(size_grid, MAX_RECTS).to(DEVICE)
    critic1 = CriticNet(size_grid, MAX_RECTS, name="critic1").to(DEVICE)
    critic2 = CriticNet(size_grid, MAX_RECTS, name="critic2").to(DEVICE)
    target_critic1 = CriticNet(size_grid, MAX_RECTS, name="target_critic1").to(DEVICE)
    target_critic2 = CriticNet(size_grid, MAX_RECTS, name="target_critic2").to(DEVICE)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=2e-4)
    critic1_optim = optim.Adam(critic1.parameters(), lr=2e-4)
    critic2_optim = optim.Adam(critic2.parameters(), lr=2e-4)

    buffer = ReplayBuffer(10000)
    BATCH_SIZE = 32
    GAMMA = 0.99
    ALPHA = 0.2 # エントロピー正則化係数
    reward_history = []
    loss_history = []
    for episode in range(num_episodes):
        rects = generate_random_rects()
        num_rects = len(rects)
        state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
        for i, (w, h) in enumerate(rects):
            rects_info[i*2:i*2+2] = [w, h]
        rects_input = np.concatenate([rects_info, [num_rects, 0.0, 0.0]]).astype(np.float32)
        rects_tensor = torch.tensor(rects_input).unsqueeze(0)
        total_reward = 0.0
        total_loss = 0.0
        for t in range(num_rects + 3):
            state_tensor = torch.tensor(state).unsqueeze(0)
            box_probs, place_probs = actor(state_tensor, rects_tensor)
            index_box, index_place = select_action(box_probs, place_probs)
            next_state, reward, success = apply_action(state, rects_info, index_box, index_place)
            count_rects = ((rects_info != 0).sum().item()) // 2
            rects_input_next = np.concatenate([rects_info.copy(), [count_rects, index_box, 0.0]]).astype(np.float32)
            next_rects_tensor = torch.tensor(rects_input_next).unsqueeze(0)
            done = not success or count_rects == 0
            buffer.push(state, rects_input, index_box, index_place, reward, next_state, rects_input_next, done)
            state = next_state
            rects_tensor = next_rects_tensor
            total_reward += reward
            if done == 0:
                break

            # 学習
            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                batch_state = torch.tensor(np.stack(transitions.state)).float().to(DEVICE)
                batch_rects = torch.tensor(np.stack(transitions.rects)).float().to(DEVICE)
                batch_index_box = torch.tensor(transitions.index_box).long().to(DEVICE)
                batch_index_place = torch.tensor(transitions.index_place).long().to(DEVICE)
                batch_reward = torch.tensor(transitions.reward).float().to(DEVICE)
                batch_next_state = torch.tensor(np.stack(transitions.next_state)).float().to(DEVICE)
                batch_next_rects = torch.tensor(np.stack(transitions.next_rects)).float().to(DEVICE)
                batch_done = torch.tensor(transitions.done).float().to(DEVICE)

                # --- Critic Update ---
                with torch.no_grad():
                    next_box_probs, next_place_probs = actor(batch_next_state, batch_next_rects)
                    next_box_logp = torch.log(next_box_probs + 1e-8)
                    next_place_logp = torch.log(next_place_probs + 1e-8)
                    q1_box, q1_place = target_critic1(batch_next_state, batch_next_rects)
                    q2_box, q2_place = target_critic2(batch_next_state, batch_next_rects)
                    min_q_box = torch.min(q1_box, q2_box)
                    min_q_place = torch.min(q1_place, q2_place)
                    v_next_box = (next_box_probs * (min_q_box - ALPHA * next_box_logp)).sum(dim=1)
                    v_next_place = (next_place_probs * (min_q_place - ALPHA * next_place_logp)).sum(dim=1)
                    target_q_box = batch_reward + (1 - batch_done) * GAMMA * v_next_box
                    target_q_place = batch_reward + (1 - batch_done) * GAMMA * v_next_place

                q1_box_pred, q1_place_pred = critic1(batch_state, batch_rects)
                q2_box_pred, q2_place_pred = critic2(batch_state, batch_rects)
                q1_box_pred = q1_box_pred.gather(1, batch_index_box.unsqueeze(1)).squeeze(1)
                q1_place_pred = q1_place_pred.gather(1, batch_index_place.unsqueeze(1)).squeeze(1)
                q2_box_pred = q2_box_pred.gather(1, batch_index_box.unsqueeze(1)).squeeze(1)
                q2_place_pred = q2_place_pred.gather(1, batch_index_place.unsqueeze(1)).squeeze(1)

                loss_critic1 = nn.MSELoss()(q1_box_pred, target_q_box) + nn.MSELoss()(q1_place_pred, target_q_place)
                loss_critic2 = nn.MSELoss()(q2_box_pred, target_q_box) + nn.MSELoss()(q2_place_pred, target_q_place)
                critic1_optim.zero_grad()
                loss_critic1.backward()
                critic1_optim.step()
                critic2_optim.zero_grad()
                loss_critic2.backward()
                critic2_optim.step()

                # --- Actor Update ---
                box_probs, place_probs = actor(batch_state, batch_rects)
                box_logp = torch.log(box_probs + 1e-8)
                place_logp = torch.log(place_probs + 1e-8)
                q1_box, q1_place = critic1(batch_state, batch_rects)
                q1_box_policy = (box_probs * (ALPHA * box_logp - q1_box)).sum(dim=1).mean()
                q1_place_policy = (place_probs * (ALPHA * place_logp - q1_place)).sum(dim=1).mean()
                loss_actor = q1_box_policy + q1_place_policy
                actor_optim.zero_grad()
                loss_actor.backward()
                actor_optim.step()
                total_loss += loss_actor.item()
                loss_history.append(total_loss)
                reward_history.append(total_reward)

                # --- ターゲットネットワークの更新 ---
                for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                    target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)
                for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                    target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)
        if episode % 100 == 0:
            actor.save_model()
            critic1.save_model()
            critic2.save_model()
            target_critic1.save_model()
            target_critic2.save_model()
        if episode % 100 == 0:
            print(f"episode {episode} total reward: {total_reward:.2f}")

    save_log(reward_history, "reward_history_reference15.txt")
    save_log(loss_history, "loss_history_reference15.txt")
    print("Training finished.")

# リストのログを保存するための関数
def save_log(log, filename):
    dir_current = os.path.dirname(os.path.abspath(__file__))
    path_log = os.path.join(dir_current, filename)
    with open(path_log, 'w') as f:
        for item in log:
            f.write(f"{item}\n")
    print(f"Log saved to {path_log}")


def eval():
    # 省略：train()と同様にActorNetで推論
    size_grid = GRID_SIZE * GRID_SIZE
    actor = ActorNet(size_grid, MAX_RECTS).to(DEVICE)
    actor.load_model()
    
    rects = generate_random_rects()
    num_rects = len(rects)
    state = np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    rects_info = np.zeros((MAX_RECTS * 2,), dtype=np.float32)
    for i, (w, h) in enumerate(rects):
        rects_info[i*2:i*2+2] = [w, h]
    rects_input = np.concatenate([rects_info, [num_rects, 0.0, 0.0]]).astype(np.float32)
    rects_tensor = torch.tensor(rects_input).unsqueeze(0)
    total_reward = 0
    for t in range(num_rects + 3):
        state_tensor = torch.tensor(state).unsqueeze(0)
        box_probs, place_probs = actor(state_tensor, rects_tensor)
        index_box, index_place = select_action(box_probs, place_probs)
        next_state, reward, success = apply_action(state, rects_info, index_box, index_place)
        print(f"Step {t+1}: index_box={index_box}, index_place={index_place}, rects_info={rects_info}")
        count_rects = ((rects_info != 0).sum().item()) // 2
        rects_input_next = np.concatenate([rects_info.copy(), [count_rects, index_box, 0.0]]).astype(np.float32)
        next_rects_tensor = torch.tensor(rects_input_next).unsqueeze(0)
        done = not success or count_rects == 0
        state = next_state
        rects_tensor = next_rects_tensor

    import matplotlib.pyplot as plt
    plt.imshow(state[0])
    plt.title('Final arrangement')
    plt.show()

if __name__ == "__main__":
    # train()
    eval()