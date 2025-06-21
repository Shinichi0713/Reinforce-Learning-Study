import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. 学習済みエージェントの定義（ダミー教師: ここでは単純なルールベース） ---
class TeacherAgent:
    def act(self, state):
        # 例: ポールが右に傾いていれば右、左なら左（本来は学習済みモデルを使う）
        return 1 if state[2] > 0 else 0

# --- 2. データ収集 ---
def collect_teacher_data(env, teacher, num_episodes=50):
    states = []
    actions = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = teacher.act(state)
            states.append(state)
            actions.append(action)
            state, _, done, _, _ = env.step(action)
    return np.array(states), np.array(actions)

# --- 3. 行動クローンモデル（模倣エージェント） ---
class BCNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# --- 4. 学習 ---
def train_bc(states, actions, obs_dim, n_actions, epochs=10):
    model = BCNet(obs_dim, n_actions)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    for epoch in range(epochs):
        logits = model(states_t)
        loss = criterion(logits, actions_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f"epoch {epoch+1}, loss {loss.item():.4f}")
    return model

# --- 5. 評価 ---
def evaluate(env, model, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_t)
                action = logits.argmax(dim=1).item()
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    print(f"模倣エージェントの平均報酬: {np.mean(total_rewards):.2f}")

# --- 実行 ---
env = gym.make("CartPole-v1")
teacher = TeacherAgent()  # 本来は学習済みモデル
states, actions = collect_teacher_data(env, teacher, num_episodes=50)
print(f"教師データ: {states.shape}, {actions.shape}")

bc_model = train_bc(states, actions, obs_dim=4, n_actions=2, epochs=10)
evaluate(env, bc_model, n_episodes=10)
