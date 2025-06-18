# エージェント学習と実行コード
import torch
import torch.optim as optim
import os
from agent import PointerNet
from environment import TSPEnv

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
def evaluate():
    # ハイパーパラメータ
    input_dim = 2
    hidden_dim = 128
    seq_len = 10
    batch_size = 8
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