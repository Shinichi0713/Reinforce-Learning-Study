import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 方策ネットワーク
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# エピソードを1つ実行して、軌跡を記録
def run_episode(env, policy_net, device):
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        probs = policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

# 割引累積報酬を計算
def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    # 標準化（学習を安定化させるため）
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# メイン処理
def main():
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

    num_episodes = 500
    gamma = 0.99
    reward_history = []

    for episode in range(num_episodes):
        states, actions, rewards = run_episode(env, policy_net, device)
        returns = compute_returns(rewards, gamma)

        loss = 0
        for logit_state, action, G in zip(states, actions, returns):
            state_tensor = torch.FloatTensor(logit_state).to(device)
            probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            loss += -log_prob * G  # REINFORCEの損失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}")

    # 学習曲線をプロット
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE on CartPole-v1')
    plt.show()

if __name__ == "__main__":
    main()
