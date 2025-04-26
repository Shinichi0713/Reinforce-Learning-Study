import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor-Critic ネットワーク
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

# 環境の設定
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ハイパーパラメータ
learning_rate = 0.01
gamma = 0.99

# モデル、オプティマイザ、損失関数の設定
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# トレーニングループ
for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    values = []
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        log_probs.append(action_dist.log_prob(action))
        values.append(state_value)
        
        next_state, reward, done, _ = env.step(action.item())
        rewards.append(reward)
        state = next_state

        if done:
            Qval = 0
            values.append(torch.FloatTensor([0]))

            # 報酬の計算
            Qvals = []
            for r in reversed(rewards):
                Qval = r + gamma * Qval
                Qvals.insert(0, Qval)

            # 損失の計算とパラメータの更新
            Qvals = torch.FloatTensor(Qvals)
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)

            advantage = Qvals - values[:-1]
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

    if episode % 100 == 0:
        print(f'Episode {episode}, Loss: {loss.item()}')

env.close()