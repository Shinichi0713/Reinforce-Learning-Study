
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import deque


def make_input(obs):
    # obs: dict with 'cities', 'visited', 'current_city'
    # 入力ベクトルは [x, y, visited_flag, is_current]
    cities = obs['cities']
    visited = obs['visited']
    current = obs['current_city']
    num_cities = cities.shape[0]
    input_vec = np.zeros((num_cities, 4), dtype=np.float32)
    input_vec[:, 0:2] = cities
    input_vec[:, 2] = visited.astype(np.float32)
    input_vec[:, 3] = 0
    input_vec[current, 3] = 1  # 現在地
    return input_vec

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
    model = PointerNet(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epochs):
        log_probs_all = []
        rewards_all = []
        tour_lengths = []

        for _ in range(batch_size):
            env = TSPEnv(num_cities=seq_len)
            obs = env.reset()
            mask = np.zeros(seq_len, dtype=bool)
            tour = []
            log_probs = []
            done = False

            for t in range(seq_len - 1):  # 最後の都市は自動的に決まる
                inp = make_input(obs)  # (N, 4)
                inp_tensor = torch.tensor(inp[None, :, :], dtype=torch.float32, device=model.device)  # (1, N, 4)
                with torch.no_grad():
                    enc_out, (h, c) = model.encoder(inp_tensor)
                dec_input = torch.zeros(1, enc_out.size(2), device=model.device)
                dec_h, dec_c = h[-1], c[-1]
                mask_torch = torch.tensor(mask[None, :], dtype=torch.bool, device=model.device)
                # すでに選択した都市ぶんデコーダを進める
                for _t in range(t):
                    idx = tour[_t]
                    dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
                    dec_input = enc_out[:, idx, :]
                # 次の都市を選ぶ
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

            log_probs_all.append(torch.stack(log_probs).sum())
            rewards_all.append(-env.total_distance)  # 報酬は距離のマイナス
            tour_lengths.append(env.total_distance)

        # ベースライン（バッチ平均）を使ったREINFORCE
        rewards_all = torch.tensor(rewards_all, dtype=torch.float32, device=model.device)
        log_probs_all = torch.stack(log_probs_all)
        baseline = rewards_all.mean()
        advantage = rewards_all - baseline
        loss = -(advantage * log_probs_all).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Avg tour length {np.mean(tour_lengths):.4f}")

        if epoch % 200 == 0:
            model.save_nn()

    print("Training complete.")
    model.save_nn()


if __name__ == "__main__":
    train_with_env()
    # model = PointerNet(input_dim, hidden_dim)
    # model.load_nn()  # モデルのロード
    # evaluate(model)  # 評価関数を呼び出す
    # evaluate_and_plot(model)  # 評価とプロットを行う
