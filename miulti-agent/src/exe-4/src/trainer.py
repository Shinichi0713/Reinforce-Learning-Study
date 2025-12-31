class Memory:
    def __init__(self):
        # 変数名を returns に統一します
        self.obs, self.states, self.actions, self.log_probs = [], [], [], []
        self.returns, self.dones, self.h_actors, self.h_critics = [], [], [], []

    def clear(self):
        self.__init__()

class MAPPOTrainer:
    def __init__(self, obs_dim, state_dim, action_dim):
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.num_agents = 2
        self.hidden_act = 128
        self.hidden_crit = 256
        
        self.actors = [GRU_Actor(obs_dim, action_dim, self.hidden_act) for _ in range(self.num_agents)]
        self.critic = GRU_Critic(state_dim, self.hidden_crit)
        
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=3e-4) for a in self.actors]
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def normalize_obs(self, obs_list, grid_size=10):
        tensors = []
        for o in obs_list:
            # 座標を0-1に正規化
            vec = [o["agent_pos"][0]/grid_size, o["agent_pos"][1]/grid_size, (o["carrying"]+1)/5.0]
            vec += [o["other_agent"][0]/grid_size, o["other_agent"][1]/grid_size]
            for p in o["packages"]:
                vec += [p[0][0]/grid_size, p[0][1]/grid_size, p[1][0]/grid_size, p[1][1]/grid_size, float(p[2]), float(p[3])]
            tensors.append(torch.FloatTensor(vec))
        return torch.stack(tensors) # (Agents, Obs_Dim)

    def train(self, memory):
        # データをテンソル化 (T, Agents, Dim)
        obs = torch.stack(memory.obs).unsqueeze(0) # (1, T, Agents, Dim)
        states = torch.stack(memory.states).unsqueeze(0) # (1, T, State_Dim)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        
        # 報酬の累積計算 (Returns)
        rewards = torch.stack(memory.returns)
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(self.num_agents)
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        # Critic更新 (GRUの初期Hiddenは保存時の最初のものを使用)
        val, _ = self.critic(states, memory.h_critics[0])
        critic_loss = F.mse_loss(val.squeeze(), returns.mean(dim=-1))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor更新
        adv = (returns.mean(dim=-1) - val.squeeze().detach())
        for i in range(self.num_agents):
            dist, _ = self.actors[i](obs[:, :, i], memory.h_actors[0][i])
            new_log_probs = dist.log_prob(actions[:, i])
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()
        
        memory.clear()