import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. エキスパートデータの収集（ここでは単純なルールベースエキスパート）
env = gym.make('CartPole-v1')

expert_obs = []
expert_acts = []

for episode in range(30):  # 30エピソード分データ収集
    obs = env.reset()
    done = False
    while not done:
        # 単純なルール: カートの角速度で制御
        action = 0 if obs[2] < 0 else 1
        expert_obs.append(obs)
        expert_acts.append(action)
        obs, reward, done, info = env.step(action)

expert_obs = np.array(expert_obs, dtype=np.float32)
expert_acts = np.array(expert_acts, dtype=np.int64)

# 2. 模倣学習用モデル（シンプルなMLP）
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)

model = PolicyNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 3. 教師あり学習で模倣学習
for epoch in range(20):
    idx = np.random.permutation(len(expert_obs))
    obs_shuffled = torch.tensor(expert_obs[idx])
    acts_shuffled = torch.tensor(expert_acts[idx])
    logits = model(obs_shuffled)
    loss = criterion(logits, acts_shuffled)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 4. 学習済みモデルでCartPole実行
obs = env.reset()
done = False
total_reward = 0
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(obs_tensor).argmax(dim=1).item()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print('Test Episode Reward:', total_reward)
env.close()
