
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import deque
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPUを使用する場合は適宜設定


def make_input(obs):
    # obs: dict from TSPEnv
    cities = obs['cities']  # (N, 2)
    visited = obs['visited'].astype(np.float32).reshape(-1, 1)  # (N, 1)
    current_city = obs['current_city']
    current_flag = np.zeros((len(cities), 1), dtype=np.float32)
    current_flag[current_city, 0] = 1.0
    # 入力: [都市座標, visited, current]
    inp = np.concatenate([cities, visited, current_flag], axis=1)  # (N, 4)
    return inp

def generate_batch(env_class, batch_size, num_cities):
    batch_inputs = []
    batch_tours = []
    batch_lens = []
    for _ in range(batch_size):
        env = env_class(num_cities)
        obs = env.reset()
        inp_seq = []
        tour = []
        done = False
        while not done:
            inp = make_input(obs)
            inp_seq.append(inp)
            # ランダム方策（模倣学習なら最適行動をここで選ぶ）
            unvisited = np.where(~obs['visited'])[0]
            if len(unvisited) == 0:
                break
            action = np.random.choice(unvisited)
            tour.append(action)
            obs, reward, done, _ = env.step(action)
        inp_seq = np.stack(inp_seq, axis=0)  # (N, num_cities, 4)
        batch_inputs.append(inp_seq)
        batch_tours.append(tour)
        batch_lens.append(env.total_distance)
    # パディング・テンソル化は省略（seq_len一定なら不要）
    return np.array(batch_inputs), np.array(batch_tours), np.array(batch_lens)

class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTMCell(hidden_dim, hidden_dim)
        self.pointer = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

        self.path_nn = os.path.join(os.path.dirname(__file__), 'path_nn.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_nn()
        self.to(self.device)

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        batch_size, seq_len, _ = inputs.size()
        enc_out, (h, c) = self.encoder(inputs)
        dec_input = torch.zeros(batch_size, enc_out.size(2)).to(inputs.device)
        dec_h, dec_c = h[-1], c[-1]
        mask = torch.zeros(batch_size, seq_len).to(inputs.device)
        indices = []
        for _ in range(seq_len):
            dec_h, dec_c = self.decoder(dec_input, (dec_h, dec_c))
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
            # maskより選択されないようにする
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            probs = torch.softmax(scores, dim=1)
            idx = probs.multinomial(1).squeeze(1)
            indices.append(idx)
            mask[torch.arange(batch_size), idx] = 1
            dec_input = enc_out[torch.arange(batch_size), idx, :]
        indices = torch.stack(indices, dim=1)
        return indices

    def save_nn(self):
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def load_nn(self):
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device))


def compute_tour_length(coords, tour_idx):
    batch_size, seq_len, _ = coords.size()
    tour = torch.gather(coords, 1, tour_idx.unsqueeze(2).expand(-1, -1, 2))
    tour_shift = torch.roll(tour, shifts=-1, dims=1)
    length = ((tour - tour_shift) ** 2).sum(2).sqrt().sum(1)
    return length

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class TSPEnv:
    def __init__(self, num_cities=10, seed=14):
        self.num_cities = num_cities
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.cities = None
        self.visited = None
        self.current_city = None
        self.total_distance = None
        self.step_count = None

    def reset(self):
        # 都市座標をランダム生成（[0,1]区間）
        self.cities = self.rng.rand(self.num_cities, 2)
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.current_city = self.rng.randint(self.num_cities)
        self.visited[self.current_city] = True
        self.total_distance = 0.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # 状態: [都市座標, 訪問フラグ, 現在地]
        return {
            'cities': self.cities.copy(),  # (num_cities, 2)
            'visited': self.visited.copy(),  # (num_cities,)
            'current_city': self.current_city
        }

    def step(self, action):
        # action: 次に移動する都市のインデックス
        if self.visited[action]:
            # 既に訪問済み都市を選んだ場合は大きなペナルティ
            reward = -100.0
            done = True
            return self._get_obs(), reward, done, {}

        # 距離計算
        prev_city = self.current_city
        dist = np.linalg.norm(self.cities[prev_city] - self.cities[action])
        self.total_distance += dist
        self.current_city = action
        self.visited[action] = True
        self.step_count += 1

        done = self.step_count == self.num_cities - 1
        reward = -dist  # 距離のマイナスを報酬

        if done:
            # 最後はスタート地点に戻る
            start_city = np.where(self.visited)[0][0]
            dist_return = np.linalg.norm(self.cities[self.current_city] - self.cities[start_city])
            self.total_distance += dist_return
            reward -= dist_return

        return self._get_obs(), reward, done, {}

    def render(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.cities[:,0], self.cities[:,1], c='blue')
        order = [np.where(self.visited)[0][0]]  # スタート地点
        for i, v in enumerate(self.visited):
            if v and i not in order:
                order.append(i)
        order.append(order[0])  # return to start
        path = self.cities[order]
        plt.plot(path[:,0], path[:,1], c='red')
        plt.title(f'Total distance: {self.total_distance:.2f}')
        plt.show()


def train_with_env():
    input_dim = 4
    hidden_dim = 128
    seq_len = 10
    batch_size = 32
    num_epochs = 10000
    buffer_capacity = 10000
    model = PointerNet(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # まずバッファを満たす
    print("Filling replay buffer...")
    while len(replay_buffer) < batch_size * 10:
        env = TSPEnv(num_cities=seq_len)
        obs = env.reset()
        mask = np.zeros(seq_len, dtype=bool)
        tour = []
        log_probs = []
        done = False

        for t in range(seq_len):
            inp = make_input(obs)
            inp_tensor = torch.tensor(inp[None, :, :], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                enc_out, (h, c) = model.encoder(inp_tensor)
            dec_input = torch.zeros(1, enc_out.size(2), device=model.device)
            dec_h, dec_c = h[-1], c[-1]
            mask_torch = torch.tensor(mask[None, :], dtype=torch.bool, device=model.device)
            for _t in range(t):
                idx = tour[_t]
                dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
                dec_input = enc_out[:, idx, :]
            dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
            scores = scores.masked_fill(mask_torch, float('-inf'))
            probs = torch.softmax(scores, dim=1)
            idx = probs.multinomial(1).item()
            log_prob = torch.log(probs[0, idx] + 1e-8)
            log_probs.append(log_prob)
            tour.append(idx)
            mask[idx] = True
            obs, reward, done, _ = env.step(idx)
            if done:
                break

        episode = {
            'log_probs': torch.stack(log_probs),
            'reward': -env.total_distance  # 報酬は距離のマイナス
        }
        replay_buffer.push(episode)

    print("Replay buffer filled.")

    for epoch in range(num_epochs):
        # 経験からサンプリング
        batch = replay_buffer.sample(batch_size)
        log_probs_all = []
        rewards_all = []

        for episode in batch:
            log_probs_all.append(episode['log_probs'].sum())
            rewards_all.append(episode['reward'])

        rewards_all = torch.tensor(rewards_all, dtype=torch.float32, device=model.device)
        log_probs_all = torch.stack(log_probs_all)
        baseline = rewards_all.mean()
        advantage = rewards_all - baseline
        loss = -(advantage * log_probs_all).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # 新しい経験も追加
        for _ in range(batch_size):
            env = TSPEnv(num_cities=seq_len)
            obs = env.reset()
            mask = np.zeros(seq_len, dtype=bool)
            tour = []
            log_probs = []
            done = False

            for t in range(seq_len):
                inp = make_input(obs)
                inp_tensor = torch.tensor(inp[None, :, :], dtype=torch.float32, device=model.device)
                with torch.no_grad():
                    enc_out, (h, c) = model.encoder(inp_tensor)
                dec_input = torch.zeros(1, enc_out.size(2), device=model.device)
                dec_h, dec_c = h[-1], c[-1]
                mask_torch = torch.tensor(mask[None, :], dtype=torch.bool, device=model.device)
                for _t in range(t):
                    idx = tour[_t]
                    dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
                    dec_input = enc_out[:, idx, :]
                dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
                scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
                scores = scores.masked_fill(mask_torch, float('-inf'))
                probs = torch.softmax(scores, dim=1)
                idx = probs.multinomial(1).item()
                log_prob = torch.log(probs[0, idx] + 1e-8)
                log_probs.append(log_prob)
                tour.append(idx)
                mask[idx] = True
                obs, reward, done, _ = env.step(idx)
                if done:
                    break

            episode = {
                'log_probs': torch.stack(log_probs),
                'reward': -env.total_distance
            }
            replay_buffer.push(episode)

        if epoch % 50 == 0:
            avg_tour_length = -rewards_all.mean().item()
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Avg tour length {avg_tour_length:.4f}")

        if epoch % 200 == 0:
            model.save_nn()

    print("Training complete.")
    model.save_nn()



def eval():
    env = TSPEnv(num_cities=10)
    model = PointerNet(input_dim=4, hidden_dim=128)
    env.reset()
    env.render()

if __name__ == "__main__":
    train_with_env()
    # eval()
