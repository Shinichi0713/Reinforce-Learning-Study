class MAPPOTrainer:
    def __init__(self, obs_dim, state_dim, action_dim):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.clip_eps = 0.2
        self.gamma = 0.98

    def train(self, memory):
        # テンソル化 (簡易版のためバッチサイズ1、全シーケンスで学習)
        obs = torch.stack(memory.obs).unsqueeze(0)        # (1, T, NumAgents, ObsDim)
        states = torch.stack(memory.states).unsqueeze(0)  # (1, T, StateDim)
        actions = torch.stack(memory.actions).unsqueeze(0) # (1, T, NumAgents)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(0)
        rewards = torch.stack(memory.rewards)             # (T, NumAgents)

        # 累積報酬 (Returns) の計算
        T = rewards.size(0)
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(2)
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        target_returns = returns.mean(dim=-1).unsqueeze(0).unsqueeze(-1) # (1, T, 1)

        # Critic 更新
        values, _ = self.critic(states, memory.h_critics[0])
        critic_loss = F.mse_loss(values, target_returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor 更新
        advantages = (target_returns - values.detach())
        actor_loss = 0
        for i in range(2):
            agent_id = torch.zeros(1, T, 2); agent_id[:, :, i] = 1.0
            actor_input = torch.cat([obs[:, :, i], agent_id], dim=-1)
            dist, _ = self.actor(actor_input, memory.h_actors[0][i])
            new_log_probs = dist.log_prob(actions[:, :, i])
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, :, i])
            surr1 = ratio * advantages.squeeze(-1)
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages.squeeze(-1)
            actor_loss += -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

# --- 実行セクション ---
env = CollaborativeDroneEnv(size=7)
trainer = MAPPOTrainer(obs_dim=7, state_dim=14, action_dim=5)
memory = MAPPOMemory()

for epi in range(1001):
    obs_list = env.reset()
    h_actors = [torch.zeros(1, 1, 128) for _ in range(2)]
    h_critic = torch.zeros(1, 1, 256)
    memory.h_actors.append([h.clone() for h in h_actors]); memory.h_critics.append(h_critic.clone())
    
    total_r = 0
    for t in range(env.max_steps):
        obs_tensor = torch.FloatTensor(np.array(obs_list))
        global_state = obs_tensor.view(-1)
        
        actions, log_probs, next_h_actors = [], [], []
        for i in range(2):
            agent_id = torch.zeros(2); agent_id[i] = 1.0
            a_input = torch.cat([obs_tensor[i], agent_id]).view(1, 1, -1)
            with torch.no_grad():
                dist, h_a = trainer.actor(a_input, h_actors[i])
                a = dist.sample()
                actions.append(a.item()); log_probs.append(dist.log_prob(a)); next_h_actors.append(h_a)
        
        next_obs, rewards, done, _ = env.step(actions)
        memory.store(obs_tensor, global_state, torch.tensor(actions), torch.tensor(log_probs), torch.FloatTensor(rewards), done)
        
        obs_list, h_actors, total_r = next_obs, next_h_actors, total_r + sum(rewards)
        if done: break
        
    trainer.train(memory)
    memory.clear()

    if epi % 100 == 0:
        print(f"Episode {epi}, Reward: {total_r:.2f}")
        # env.save_gif(filename=f"mappo_coop_ep{epi}.gif")