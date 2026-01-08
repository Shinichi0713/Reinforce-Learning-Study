class MAPPOTrainer_V2:
    def __init__(self, obs_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.eps = 1e-8
        
        # 工夫1: パラメータ共有Actor (+ID次元)
        self.actor = GRU_Actor(obs_dim + num_agents, action_dim, 128)
        # 工夫2: 集中Critic (全員の観測を結合)
        self.critic = GRU_Critic(obs_dim * num_agents, 256)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def train(self, memory):
        # テンソル化 (T, NumAgents, Dim)
        obs = torch.stack(memory.obs) 
        states = torch.stack(memory.states).unsqueeze(0) # Critic用 (1, T, StateDim)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        rewards = torch.stack(memory.rewards)
        
        T = obs.size(0)

        # --- 累積報酬 (Returns) の計算 ---
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(self.num_agents)
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # --- Critic の更新 (集中学習) ---
        values, _ = self.critic(states, memory.h_critics[0])
        values = values.squeeze() # (T, 1) -> (T)
        # チーム全体の平均リターンを予測対象とする（または各リターンを結合）
        target_returns = returns.mean(dim=-1)
        critic_loss = F.mse_loss(values, target_returns)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # 勾配クリップ
        self.critic_opt.step()

        # --- Actor の更新 (パラメータ共有学習) ---
        # 全エージェントのデータを1つのバッチにまとめる
        advantages = (target_returns - values.detach()).unsqueeze(-1).repeat(1, self.num_agents)
        
        actor_loss_total = 0
        for i in range(self.num_agents):
            # Agent ID の付与
            ids = torch.zeros(T, self.num_agents)
            ids[:, i] = 1.0
            combined_obs = torch.cat([obs[:, i], ids], dim=-1).unsqueeze(0) # (1, T, Obs+ID)
            
            # 再計算
            dist, _ = self.actor(combined_obs, memory.h_actors[0][i])
            new_log_probs = dist.log_prob(actions[:, i])
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            surr1 = ratio * advantages[:, i]
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[:, i]
            
            actor_loss_total += -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

        self.actor_opt.zero_grad()
        actor_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 勾配クリップ
        self.actor_opt.step()

        memory.clear()