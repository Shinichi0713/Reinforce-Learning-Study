import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from google.colab import drive
import os

# ハイパーパラメータ
GAMMA = 0.99            # 割引率
LR = 0.0005             # 学習率
ENTROPY_BETA = 0.01     # 探索を促すエントロピー項の係数
NUM_EPISODES = 1000     # 学習エピソード数

drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_go"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    env = GoEnv()
    model = GoPolicyValueNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False
        
        log_probs = []
        values = []
        rewards = []
        entropies = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) # (1, 3, 19, 19)
            policy_logits, value = model(state_tensor)

            # 非合法手をマスク (非合法手のlogitを極小値にして確率0化)
            legal_mask = env.get_legal_actions()
            mask_tensor = torch.FloatTensor(legal_mask).to(device)
            masked_logits = policy_logits - (1.0 - mask_tensor) * 1e9

            # 行動の確率分布を作成
            probs = torch.softmax(masked_logits, dim=-1)
            dist = Categorical(probs)

            # 行動をサンプリング
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # 環境のステップを実行
            next_state, reward, done, _, _ = env.step(action.item())

            # ログの保持
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state

        # --- リターン（累積割引報酬）とアドバンテージの計算 ---
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze(-1)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # Advantage = Return - Value
        advantages = returns - values.detach()

        # 損失の計算 (Actor Loss + Critic Loss - Entropy)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + 0.5 * critic_loss + ENTROPY_BETA * entropy_loss

        # 逆伝播と最適化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode:4d} | Total Reward: {total_reward:6.1f} | Loss: {total_loss.item():.4f}")

    # 学習済みモデルをONNX形式でエクスポート
    export_to_onnx(model, os.path.join(CHECKPOINT_DIR, "go_agent_trained.onnx"))

if __name__ == "__main__":
    train()