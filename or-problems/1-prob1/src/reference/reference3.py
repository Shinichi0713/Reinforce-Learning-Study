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

    def compute_tour_length(self, tour_idx):
        # batch_size, seq_len, _ = self.cities.size()
        tour = torch.gather(self.cities.to(tour_idx.device), 1, tour_idx.unsqueeze(2).expand(-1, -1, 2))
        tour_shift = torch.roll(tour, shifts=-1, dims=1)
        length = ((tour - tour_shift) ** 2).sum(2).sqrt().sum(1)
        return length

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


def train():
    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 64
    model = PointerNet(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(10):
        for epoch in range(1000):
            env = TSPEnv(batch_size, seq_len)
            coords = env.reset()
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
            # loss = -(reward * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 勾配クリッピング
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss {loss.item():.4f}, Avg tour length {tour_len.mean().item():.4f}")
        # モデルの保存
        model.save_nn()
    print("Training complete.")


def evaluate(model, num_batches=100, batch_size=64, seq_len=10):
    model.eval()
    total_length = 0
    total_samples = 0

    with torch.no_grad():
        for _ in range(num_batches):
            coords = torch.rand(batch_size, seq_len, 2).to(model.device)
            tour_idx = model(coords)
            tour_len = compute_tour_length(coords, tour_idx)
            total_length += tour_len.sum().item()
            total_samples += batch_size

    avg_length = total_length / total_samples
    print(f"Evaluation: Average tour length over {total_samples} samples: {avg_length:.4f}")
    return avg_length

def evaluate_and_plot(model, num_batches=100, batch_size=64, seq_len=10):
    model.eval()
    all_lengths = []

    with torch.no_grad():
        for _ in range(num_batches):
            coords = torch.rand(batch_size, seq_len, 2).to(model.device)
            tour_idx = model(coords)
            tour_len = compute_tour_length(coords, tour_idx)
            all_lengths.extend(tour_len.cpu().numpy())

    avg_length = sum(all_lengths) / len(all_lengths)
    print(f"Evaluation: Average tour length over {len(all_lengths)} samples: {avg_length:.4f}")

    # 可視化: ツアー長のヒストグラム
    plt.hist(all_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Tour Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tour Lengths')
    plt.show()

    return avg_length, all_lengths

def plot_sample_tour(model, seq_len=10):
    model.eval()
    with torch.no_grad():
        coords = torch.rand(1, seq_len, 2).to(model.device)
        tour_idx = model(coords)
        coords_np = coords.squeeze(0).cpu().numpy()
        tour_idx_np = tour_idx.squeeze(0).cpu().numpy()
        tour_coords = coords_np[tour_idx_np]

        plt.figure(figsize=(6, 6))
        plt.scatter(coords_np[:, 0], coords_np[:, 1], c='red', label='Cities')
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], '-o', label='Tour')
        # 始点と終点をつなぐ
        plt.plot([tour_coords[-1, 0], tour_coords[0, 0]], [tour_coords[-1, 1], tour_coords[0, 1]], 'o-', c='blue')
        plt.title('Sample Tour')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    train()
    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 64

    model = PointerNet(input_dim, hidden_dim)
    path_nn = os.path.join(os.path.dirname(__file__), 'path_nn.pt')
    model.load_state_dict(torch.load(path_nn, map_location=model.device))
    model.to(model.device)

    avg_length, all_lengths = evaluate_and_plot(model, num_batches=100, batch_size=64, seq_len=10)
    plot_sample_tour(model, seq_len=10)