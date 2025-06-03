

import torch
import numpy as np
import random, os
import environment
from agent import DQNNetwork
from collections import deque

N_DISCRETE_ACTIONS = 101
ACTION_SPACE = np.linspace(-2.0, 2.0, N_DISCRETE_ACTIONS)

# 経験再生バッファ
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)


def select_action(q_net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTION_SPACE))
    else:
        state_v = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_net(state_v)
        return torch.argmax(q_values).item()

def write_log(file_path, data):
    with open(file_path, 'a') as f:
        f.write(data + '\n')


def soft_update(target_net, online_net, tau=0.005):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

# DQNのトレーニング関数
def train():
    env = environment.Environment()
    state_dim = env.env.observation_space.shape[0]
    action_dim = len(ACTION_SPACE)

    # DQNネットワークの初期化
    dqn_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim)
    dqn_net.train()
    target_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim)
    target_net.eval()
    optimizer = torch.optim.AdamW(dqn_net.parameters(), lr=1e-3)

    # 経験再生バッファの初期化
    buffer = ReplayBuffer(capacity=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.97

    step_episode = 200
    count_train = 0.0
    interval_update = 1

    num_episodes = 1000
    loss_history = []
    reward_avr_history = []
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        loss_total = 0

        for step in range(step_episode):
            id_action = select_action(dqn_net, state, epsilon)
            action = np.array([ACTION_SPACE[id_action]])
            state_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # 経験をバッファに保存
            buffer.push(state, id_action, reward, state_, done)
            state = state_
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_v = torch.FloatTensor(states).to(dqn_net.device)
                actions_v = torch.LongTensor(actions).to(dqn_net.device)
                rewards_v = torch.FloatTensor(rewards).to(dqn_net.device)
                next_states_v = torch.FloatTensor(next_states).to(dqn_net.device)
                dones_v = torch.FloatTensor(dones).to(dqn_net.device)
                actions_tensor = actions_v.unsqueeze(1)  # shape: [batch_size, 1]
                q_values = dqn_net(states_v)               # shape: [batch_size, action_dim]
                q_values = q_values.gather(1, actions_tensor).squeeze(1)  # shape: [batch_size]
                with torch.no_grad():
                    next_q_values = target_net(next_states_v).max(1)[0]   # shape: [batch_size, action_dim]
                    target_q_values = rewards_v + (gamma * next_q_values * (1 - dones_v))

                loss = torch.nn.functional.mse_loss(q_values, target_q_values)
                loss_total += loss.item()
                count_train += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                soft_update(target_net, dqn_net, tau=0.005)

            if done:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            dqn_net.save_model()
        loss_history.append(loss_total / count_train)
        reward_avr_history.append(total_reward / step_episode)
        # Epsilonの更新
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print("Training completed.")
    env.close()

    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(os.path.join(dir_current, "loss_history.txt"), str(loss_history))
    write_log(os.path.join(dir_current, "reward_history.txt"), str(reward_avr_history))

# エージェントを稼働
def run_agent():
    env = environment.Environment('human')  # 'human'モードでレンダリング
    dqn_net = DQNNetwork(input_dim=env.env.observation_space.shape[0], output_dim=len(ACTION_SPACE))
    dqn_net.eval()  # 評価モードに設定
    dqn_net.to(dqn_net.device)

    state, info = env.reset()
    total_reward = 0
    for step in range(200):
        id_action = select_action(dqn_net, state, epsilon=0.0)  # Epsilonを0にして最適行動を選択
        action = np.array([ACTION_SPACE[id_action]])
        state_, reward, terminated, truncated, info = env.step(action)
        state = state_
        total_reward += reward
        env.render()
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

        if terminated or truncated:
            break

    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train()
    run_agent()

