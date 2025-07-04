import os
import torch
import numpy as np
from agent import Actor, Critic, ReplayBuffer
from environment import Environment
import torch.nn.functional as F
import copy

class Trainer:
    def __init__(self, max_action, min_action):
        self.env = Environment(is_train=True)
        state_dim, action_dim = self.env.get_dimensions()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.max_action = max_action
        self.min_action = min_action
        self.replay_buffer = ReplayBuffer(size_max=1000000, batch_size=64)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.total_it = 0

    def save_models(self):
        self.actor.save()
        self.critic.save()

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.actor.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, self.min_action, self.max_action)

    def train_once(self, batch_size=64, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1

        # サンプル
        state, action, reward, next_state, done = self.replay_buffer.sample()
        state = torch.FloatTensor(state).to(self.actor.device)
        action = torch.FloatTensor(action).to(self.actor.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.actor.device)
        next_state = torch.FloatTensor(next_state).to(self.actor.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.actor.device)

        # ターゲットアクションにノイズを加える（ターゲットポリシースムージング）
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

        # ターゲットQ値
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * gamma * target_Q.detach()

        # 現在のQ値
        current_Q1, current_Q2 = self.critic(state, action)

        # クリティック損失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # クリティックの更新 + 勾配クリッピング
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # アクターの更新（delayed policy update）
        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # ターゲットネットワークのsoft update
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_td3(self, episodes=10000, start_timesteps=0, expl_noise=0.1, reward_scale=1.0):
        expl_noise_initial = expl_noise
        expl_noise_final = 0.01
        expl_noise_decay = (expl_noise_initial - expl_noise_final) / (episodes * 0.5)
        expl_noise = expl_noise_initial

        for ep in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                if self.total_it < start_timesteps:
                    action = np.random.uniform(self.min_action, self.max_action, size=self.action_dim)
                else:
                    action = self.select_action(state, noise=expl_noise)
                next_state, reward, done, _ = self.env.step(action)
                reward /= reward_scale  # リワード正規化
                self.replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state
                episode_reward += reward

                # 学習
                self.train_once()

            # ノイズ減衰
            if expl_noise > expl_noise_final:
                expl_noise -= expl_noise_decay

            if ep % 10 == 0:
                self.save_models()
                print(f"Episode {ep + 1}/{episodes}, Reward: {episode_reward}")

def eval():
    env = Environment(is_train=False)
    max_action=1.0
    state_dim, action_dim = env.get_dimensions()
    actor = Actor(state_dim, action_dim, max_action)
    actor.eval()
    # 評価のためのコードをここに追加
    for ep in range(10):
        state = env.reset()
        episode_reward = 0
        done = False
        with torch.no_grad():
            while not done:
                state = torch.FloatTensor(state.reshape(1, -1)).to(actor.device)
                action = actor(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                state = next_state
                episode_reward += reward
                env.render()
        print(f"Episode {ep + 1}, Reward: {episode_reward}")
    env.close()

if __name__ == "__main__":
    # trainer = Trainer(max_action=1.0, min_action=-1.0)
    # trainer.train_td3()
    print("Training completed.")

    eval()
    print("Evaluation completed.")

