import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

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


# TSP環境クラス
class TSPEnv:
    def __init__(self, batch_size, num_cities=10):
        self.batch_size = batch_size
        self.num_cities = num_cities
        self.cities = None
        self.visited = None

    def reset(self):
        # 都市座標をランダム生成（[0,1]区間）
        self.cities = np.random.rand(self.batch_size, self.num_cities, 2)
        self.cities = torch.tensor(self.cities, dtype=torch.float32)
        self.visited = np.zeros((self.batch_size, self.num_cities), dtype=bool)
        self.current_city = np.random.randint(self.num_cities)
        self.visited[np.arange(self.batch_size), self.current_city] = True
        return self.cities

    # 巡回経路の長さを計算
    # 長さの負の数を報酬と定義
    def compute_tour_length(self, tour_idx):
        tour = torch.gather(self.cities.to(tour_idx.device), 1, tour_idx.unsqueeze(2).expand(-1, -1, 2))
        tour_shift = torch.roll(tour, shifts=-1, dims=1)
        length = ((tour - tour_shift) ** 2).sum(2).sqrt().sum(1)
        return length

    # エージェントが選択した順序をもとに描画
    def render(self, tour_idx):
        """
        tour_idx: (batch_size, seq_len) 各バッチごとの巡回都市順インデックス
        """
        ncols = min(4, self.batch_size)
        nrows = (self.batch_size + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)  # 1次元配列化

        for b in range(self.batch_size):
            ax = axes[b]
            cities_b = self.cities[b].cpu().numpy()  # (num_cities, 2)
            order = tour_idx[b].cpu().numpy()        # (seq_len,)
            path = cities_b[order]
            # 都市を点で描画
            ax.scatter(cities_b[:, 0], cities_b[:, 1], c='blue')
            # 巡回経路を線で描画
            ax.plot(path[:, 0], path[:, 1], c='red', marker='o')
            # スタートとゴールを強調
            ax.scatter(path[0, 0], path[0, 1], c='green', s=100, label='Start')
            ax.scatter(path[-1, 0], path[-1, 1], c='orange', s=100, label='End')
            # 始点と終点をつなぐ
            ax.plot([path[-1, 0], path[0, 0]], [path[-1, 1], path[0, 1]], 'o-', c='blue')
            ax.set_title(f'Batch {b}')
            ax.legend()

        # 不要なsubplotを消す
        for b in range(self.batch_size, nrows * ncols):
            fig.delaxes(axes[b])

        plt.tight_layout()
        plt.show()

# Reinforcement Learningのトレーニングループ
# Pointer Networkを用いてTSPを解く
def train():
    # ハイパーパラメータ
    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 64
    # モデルとオプティマイザの初期化
    model = PointerNet(input_dim, hidden_dim)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # トレーニングループ
    reward_history = []
    loss_history = []
    for epoch in range(10000):
        env = TSPEnv(batch_size, seq_len)
        coords = env.reset()
        # エージェントに巡回経路を計算させる
        tour_idx = model(coords)
        tour_len = env.compute_tour_length(tour_idx.to(model.device))
        reward = -tour_len  # 距離が短いほど報酬が高い
        baseline = reward.mean()
        advantage = reward - baseline
        # log_prob計算
        enc_out, (h, c) = model.encoder(coords.to(model.device))
        dec_input = torch.zeros(batch_size, enc_out.size(2)).to(model.device)
        dec_h, dec_c = h[-1], c[-1]
        mask = torch.zeros(batch_size, seq_len).to(model.device)
        log_probs = []
        for t in range(seq_len):
            dec_h, dec_c = model.decoder(dec_input, (dec_h, dec_c))
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
            probs = torch.softmax(scores, dim=1)
            idx = tour_idx[:, t]
            log_prob = torch.log(probs[torch.arange(batch_size), idx] + 1e-8)
            log_probs.append(log_prob)
            mask[torch.arange(batch_size), idx] = 1
            dec_input = enc_out[torch.arange(batch_size), idx, :]
        log_probs = torch.stack(log_probs, dim=1).sum(1)

        loss = -(advantage * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 勾配クリッピング。学習の安定化のため
        optimizer.step()
        # ログの保存
        reward_history.append(reward.mean().item())
        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Avg tour length {tour_len.mean().item():.4f}")
    # モデルの保存
    model.save_nn()
    # ログの保存
    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(os.path.join(dir_current, "reward_history.txt"), str(reward_history))
    write_log(os.path.join(dir_current, "loss_history.txt"), str(loss_history))
    print("Training complete.")

def write_log(file_path, data):
    with open(file_path, 'a') as f:
        f.write(data + '\n')

# モデルを使って実際にTSPを解く
def evaluate(batch_size=8, seq_len=10):
    input_dim = 2
    hidden_dim = 128
    model = PointerNet(input_dim, hidden_dim)
    model.eval()

    with torch.no_grad():
        env = TSPEnv(batch_size, seq_len)
        coords = env.reset()
        tour_idx = model(coords)
        env.render(tour_idx)


if __name__ == "__main__":
    train()
    evaluate()
