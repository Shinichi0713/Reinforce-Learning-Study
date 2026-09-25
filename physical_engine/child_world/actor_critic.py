import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- 1. Actor-Critic 統合ネットワーク ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 特徴抽出用の共通バックボーン
        self.fc1 = nn.Linear(state_dim, 128)
        
        # Actor: 行動の確率分布を出力 (Categorical Distribution)
        self.actor = nn.Linear(128, action_dim)
        
        # Critic: 状態価値 V(s) を出力
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


# --- 2. エージェントクラス ---
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.network(state_tensor)
        
        # カテゴリカル分布から行動をサンプリング
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, value.squeeze(0), entropy

    def update(self, log_prob, value, next_state, reward, done, entropy):
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # 次状態の価値 V(s_{t+1})
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.squeeze(0)
            
            # TD Target: r + gamma * V(s_{t+1}) * (1 - done)
            target_value = reward + (1.0 - float(done)) * self.gamma * next_value

        # TD Error (Advantage A(s, a) = R - V(s))
        advantage = target_value - value

        # 損失関数の計算
        # 1. Actor Loss: - log_prob * Advantage - entropy_bonus
        actor_loss = -log_prob * advantage.detach() - self.entropy_coef * entropy
        
        # 2. Critic Loss: MSE(V(s), Target)
        critic_loss = F.mse_loss(value, target_value)
        
        # Total Loss
        total_loss = actor_loss + critic_loss

        # 逆伝播と最適化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


# --- 3. 学習ループ (CartPole-v1) ---
def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)
    num_episodes = 500

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 1ステップごとのオンライン更新 (1-step TD)
            agent.update(log_prob, value, next_state, reward, done, entropy)

            state = next_state
            total_reward += reward

        if episode % 50 == 0:
            print(f"Episode {episode:3d} | Total Reward: {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    main()