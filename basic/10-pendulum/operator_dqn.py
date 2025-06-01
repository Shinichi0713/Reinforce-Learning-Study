
import torch
import numpy as np
import random, os
import environment
from agent import DQNNetwork


N_DISCRETE_ACTIONS = 21
ACTION_SPACE = np.linspace(-2.0, 2.0, N_DISCRETE_ACTIONS)

# 経験再生バッファ
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_, done))
        else:
            self.buffer[self.position] = (s, a, r, s_, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def select_action(q_net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTION_SPACE))
    else:
        state_v = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_net(state_v)
        return torch.argmax(q_values).item()

def train():
    env = environment.Environment()
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]

    # DQNネットワークの初期化
    dqn_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim)
    optimizer = torch.optim.AdamW(dqn_net.parameters(), lr=1e-3)

    # 経験再生バッファの初期化
    buffer = ReplayBuffer(capacity=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    num_episodes = 1000

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0

        for step in range(200):
            id_action = select_action(dqn_net, obs, epsilon)
            action = np.array([ACTION_SPACE[id_action]])
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # 経験をバッファに保存
            buffer.push(obs, action, reward, obs_, done)
            obs = obs_
            total_reward += reward

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(states, dtype=torch.float32).to(dqn_net.device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(dqn_net.device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(dqn_net.device)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(dqn_net.device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(dqn_net.device)
                actions_tensor = actions_tensor.squeeze(1)  # shape: [batch_size, 1]
                q_values = dqn_net(states_tensor)             # shape: [batch_size, action_dim]
                q_values = q_values.gather(1, actions_tensor) # shape: [batch_size, 1]
                q_values = q_values.squeeze(1)                # shape: [batch_size]
                with torch.no_grad():
                    next_q_values = dqn_net(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

                loss = torch.nn.functional.mse_loss(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            dqn_net.save_model()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print("Training completed.")
    env.close()


if __name__ == "__main__":
    train()
    

