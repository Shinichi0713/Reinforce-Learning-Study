import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTMCell(hidden_dim, hidden_dim)
        self.pointer = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
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

    for epoch in range(1000):
        coords = torch.rand(batch_size, seq_len, 2)
        tour_idx = model(coords)
        tour_len = compute_tour_length(coords, tour_idx)
        reward = -tour_len  # 距離が短いほど報酬が高い

        # log_prob計算
        enc_out, (h, c) = model.encoder(coords)
        dec_input = torch.zeros(batch_size, enc_out.size(2))
        dec_h, dec_c = h[-1], c[-1]
        mask = torch.zeros(batch_size, seq_len)
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

if __name__ == "__main__":
    train()
    print("Training complete.")
