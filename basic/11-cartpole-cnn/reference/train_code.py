import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque, namedtuple
from torchvision import transforms

def get_screen(env):
    # 環境からRGB画像を取得（render_mode='rgb_array'が必要）
    screen = env.render()  # ここでscreenに画像が入る
    # 画像がNoneでないことを確認
    assert screen is not None, "env.render() returned None. Check render_mode."
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((40, 80)),
        transforms.ToTensor()
    ])
    screen = transform(screen)
    return screen.unsqueeze(0)


class CNN_DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # 計算してサイズを合わせる
        dummy_input = torch.zeros(1, 1, 40, 80)
        n_flatten = self.conv(dummy_input).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self.path_nn = os.path.join(os.path.dirname(__file__), 'cnn_dqn.pth')
        self.__load_state_dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.fc(x)
        return x

    def save_to_state_dict(self):
        torch.save(self.state_dict(), self.path_nn)

    def __load_state_dict(self):
        if os.path.exists(self.path_nn):
            print(f"Loading model from {self.path_nn}")
            state_dict = torch.load(self.path_nn)
            self.load_state_dict(state_dict)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, n_actions, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q_values = policy_net(state)
        return q_values.argmax(1).item()


def train():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset()
    n_actions = env.action_space.n
    policy_net = CNN_DQN(n_actions)
    target_net = CNN_DQN(n_actions)
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10000)
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON = 0.8

    for episode in range(100):
        state = get_screen(env)
        total_reward = 0
        env.reset()
        for t in range(200):
            action = select_action(state, policy_net, n_actions, EPSILON)
            EPSILON *= 0.99  # ε-greedyのεを減衰
            _, reward, done, truncated, info= env.step(action)
            next_state = get_screen(env)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            # 学習
            # print(len(buffer))
            if len(buffer) >= BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                batch_state = torch.cat(transitions.state).to(policy_net.device)
                batch_action = torch.tensor(transitions.action).unsqueeze(1).to(policy_net.device)
                batch_reward = torch.tensor(transitions.reward, dtype=torch.float32).to(policy_net.device)
                batch_next_state = torch.cat(transitions.next_state).to(policy_net.device)
                batch_done = torch.tensor(transitions.done, dtype=torch.bool).to(policy_net.device)
                q_values = policy_net(batch_state).gather(1, batch_action).squeeze()
                with torch.no_grad():
                    max_next_q = target_net(batch_next_state).max(1)[0]
                    target = batch_reward + GAMMA * max_next_q * (~batch_done)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # ターゲットネットの更新
        if episode % 10 == 0:
            policy_net.save_to_state_dict()
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total reward: {total_reward}")
    env.close()

def evaluate():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    num_actions = env.action_space.n

    policy_net = CNN_DQN(num_actions)

    num_episodes = 5
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            screen = get_screen(env)  # (1, 1, 40, 80)
            with torch.no_grad():
                logits = policy_net(screen)
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode}: Total reward = {total_reward}")

    env.close()
if __name__ == "__main__":
    train()
    evaluate()
