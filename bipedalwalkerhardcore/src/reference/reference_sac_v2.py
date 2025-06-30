import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- ReplayBufferはご提示のものを使ってください ---
class ReplayBuffer:
    """経験再生用のバッファークラス"""
    def __init__(self, size_max=5000, batch_size=64):
        # バッファーの初期化
        self.buffer = deque(maxlen=size_max)
        self.batch_size = batch_size

    def add(self, experience):
        state, action, reward, next_state, done = experience
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = float(reward)
        done = float(done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        # idx = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        # return [self.buffer[ii] for ii in idx]
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
# SAC用アクターネットワーク
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.mid_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_actor_sac.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_state_dict()
        self.to(self.device)

    def forward(self, state):
        x = self.mid_layer(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    # SACを確率論的な分布に基づき行動選択
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def save(self):
        """モデルのパラメータを保存"""
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self, strict=False, assign=False):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device), strict=strict)
            print('...actor network loaded...')

# Vネットワークの追加
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_value_sac.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_state_dict()
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        v = self.v(x)
        return v

    def save(self):
        """モデルのパラメータを保存"""
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self, strict=False, assign=False):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device), strict=strict)
            print('...value network loaded...')


# SAC用クリティックネットワーク
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.mid_layer_q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.fc_q1 = nn.Linear(256, 1)
        # Q2
        self.mid_layer_q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.fc_q2 = nn.Linear(256, 1)

        self.path_nn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_critic_sac.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_state_dict()
        self.to(self.device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.mid_layer_q1(sa)
        q1 = self.fc_q1(q1)

        q2 = self.mid_layer_q2(sa)
        q2 = F.relu(self.fc_q2(q2))
        return q1, q2

    def save(self):
        """モデルのパラメータを保存"""
        self.cpu()
        torch.save(self.state_dict(), self.path_nn)
        self.to(self.device)

    def __load_state_dict(self, strict=False, assign=False):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location=self.device), strict=strict)
            print('...actor network loaded...')

# SACエージェント
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.value_target = ValueNetwork(state_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.value_target.load_state_dict(self.value_net.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

        self.device = device
        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # --- 1. Valueネットワークの更新 ---
        with torch.no_grad():
            new_action, new_log_prob = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, new_action)
            min_q_new = torch.min(q1_new, q2_new)
            v_target = min_q_new - torch.exp(self.log_alpha) * new_log_prob

        v = self.value_net(state)
        value_loss = F.mse_loss(v, v_target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # --- 2. Criticネットワークの更新 ---
        with torch.no_grad():
            next_v = self.value_target(next_state)
            target_q = reward + (1 - done) * self.gamma * next_v

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # アクターの更新
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        v = self.value_net(state).detach()
        advantage = torch.min(q1_new, q2_new) - v
        actor_loss = (torch.exp(self.log_alpha) * log_prob - advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 4. Alphaの更新 ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 5. ターゲットネットワークのソフトアップデート ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.value_net.parameters(), self.value_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss, alpha_loss

    def save_models(self):
        self.actor.save()
        self.critic.save()

# --- メイン学習ループ ---

def main():
    env = gym.make('BipedalWalkerHardcore-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(size_max=1000000, batch_size=256)

    episodes = 1000
    start_timesteps = 10000
    batch_size = 256
    episode_rewards = []
    mode_exploration = True  # 初期は探索モード
    total_steps = 0

    loss_actor_history = []
    loss_critic_history = []
    loss_alpha_history = []
    for episode in range(episodes):
        # reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_loss_actor = 0
        episode_loss_critic = 0
        episode_loss_alpha = 0
        count_episode = 1e-4
        done = False
        while not done:
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                mode_exploration = False  # 探索モードを終了
            action = np.clip(action, env.action_space.low, env.action_space.high)
            # actionの選択
            # action = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            next_state = np.array(next_state, dtype=np.float32)

            # bufferに格納
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= start_timesteps and len(replay_buffer) > batch_size:
                actor_loss, critic_loss, alpha_loss = agent.train(replay_buffer, batch_size)
                episode_loss_actor += actor_loss.item()
                episode_loss_critic += critic_loss.item()
                episode_loss_alpha += alpha_loss.item()
                count_episode += 1
                
        episode_rewards.append(episode_reward)
        loss_actor_history.append(episode_loss_actor / count_episode)
        loss_critic_history.append(episode_loss_critic / count_episode)
        loss_alpha_history.append(episode_loss_alpha / count_episode)

        # 進捗の表示(探索モードかではないかはわかるようにする)
        if mode_exploration:
            print(f"Exploration mode: Episode {episode}, Reward: {episode_reward}")
        else:
            print(f"Episode {episode}, Reward: {episode_reward}")

        # 100エピソードごとに平均報酬を表示
        if (episode + 1) % 100 == 0:
            print(f"Last 100 episodes average reward: {np.mean(episode_rewards[-100:])}")
            agent.save_models()

    # ログファイルに書き込む
    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(os.path.join(dir_current, 'loss_sac_actor.txt'), loss_actor_history)
    write_log(os.path.join(dir_current, 'loss_sac_critic.txt'), loss_critic_history)
    write_log(os.path.join(dir_current, 'loss_sac_alpha.txt'), loss_alpha_history)
    write_log(os.path.join(dir_current, 'reward_sac.txt'), episode_rewards)

# ログファイルに書き込む関数
def write_log(file_path, data):
    with open(file_path, 'w' , encoding='utf-8') as f:
        f.writelines([f"{item}\n" for item in data])

if __name__ == "__main__":
    main()
