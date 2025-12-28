import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 各エージェントの方策ネットワーク
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.fc(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2) # 数値安定化
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample() # reparameterization trick
        # 離散アクションの場合は Gumbel-Softmax 等を使用
        return x, dist.log_prob(x).sum(dim=-1, keepdim=True)

# チーム全体のQ値を評価するネットワーク
class Critic(nn.Module):
    def __init__(self, state_dim, all_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # 全員の状態と全員の行動を結合して入力
        self.fc = nn.Sequential(
            nn.Linear(state_dim + all_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.fc(x)


    def update_actors(self, batch):
        states, obs_list, actions_list, rewards, next_states, dones = batch
        
        # 全エージェントのアクションを最新の方策でサンプリングし直す（逐次更新の準備）
        current_actions = []
        current_log_probs = []
        for i in range(self.n_agents):
            action, log_prob = self.actors[i].sample(obs_list[i])
            current_actions.append(action)
            current_log_probs.append(log_prob)

        # 各エージェントごとに順番にロスを計算
        for i in range(self.n_agents):
            # 他のエージェントの行動を固定し、エージェントiの行動だけを微分対象にする
            joint_actions = torch.cat([
                # 0〜i-1までは新しい行動、iは現在の微分対象、i+1以降はサンプリング時の行動
                # (簡易的には全員新しい行動を入れても学習は回りますが、HASACの厳密解は逐次です)
                *current_actions[:i+1], *actions_list[i+1:]
            ], dim=-1)
            
            q_values = self.critic(states, joint_actions)
        
        # SACの目的関数: Q値 + α * エントロピー
        actor_loss = (self.alpha * current_log_probs[i] - q_values).mean()
        
        self.actor_optimizers[i].zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizers[i].step()