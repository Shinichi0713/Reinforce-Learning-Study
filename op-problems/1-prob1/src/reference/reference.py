import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

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


def train():
    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 64
    model = PointerNet(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(10):
        for epoch in range(1000):
            coords = torch.rand(batch_size, seq_len, 2)
            tour_idx = model(coords)
            tour_len = compute_tour_length(coords.to(model.device), tour_idx.to(model.device))
            reward = -tour_len  # 距離が短いほど報酬が高い

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

            loss = -(reward * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss {loss.item():.4f}, Avg tour length {tour_len.mean().item():.4f}")
        # モデルの保存
        model.save_nn()


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


if __name__ == "__main__":
    train()
    print("Training complete.")

    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 64

    model = PointerNet(input_dim, hidden_dim)
    path_nn = os.path.join(os.path.dirname(__file__), 'path_nn.pt')
    model.load_state_dict(torch.load(path_nn, map_location=model.device))
    model.to(model.device)

    evaluate(model, num_batches=100, batch_size=batch_size, seq_len=seq_len)