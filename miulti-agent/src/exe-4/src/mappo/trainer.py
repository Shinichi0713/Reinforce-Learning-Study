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
        if len(memory.obs) < 2: return
        
        # データのテンソル化
        # memory.obs: [T, Agents, Dim] -> (batch=1, T, Agents, Dim)
        obs = torch.stack(memory.obs).unsqueeze(0) 
        # memory.states: [T, State_Dim] -> (batch=1, T, State_Dim)
        states = torch.stack(memory.states).unsqueeze(0)
        actions = torch.stack(memory.actions) # (T, Agents)
        rewards = torch.stack(memory.returns) # (T, Agents)
        
        T = obs.size(1)
        
        # --- Critic 更新 ---
        with torch.no_grad():
            # 次のステップのアクション確率を計算
            next_probs = []
            for i in range(self.num_agents):
                # 最初のステップのHiddenを渡してシーケンス全体を一気に処理
                p, _ = self.actors[i](obs[:, 1:, i], memory.h_actors[0][i])
                next_probs.append(p)
            
            # (batch, T-1, Agents * Action_Dim) に整形
            next_actions_onehot = torch.cat(next_probs, dim=-1)
            
            # ターゲットQ値
            target_q1 = self.target_critics[0](states[:, 1:], next_actions_onehot)
            target_q2 = self.target_critics[1](states[:, 1:], next_actions_onehot)
            target_q = torch.min(target_q1, target_q2)
            
            # ソフト状態価値の計算 (Entropyを考慮)
            # 全エージェントの平均的なEntropyを引く
            current_entropy = 0
            for p in next_probs:
                current_entropy += -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)
            
            y = rewards[:-1].mean(dim=-1, keepdim=True) + self.gamma * (target_q.squeeze(-1) + torch.exp(self.log_alphas[0]) * current_entropy.squeeze(-1))

        # 現在のQ値の計算
        # 実際に取った行動をOne-hot化して結合
        curr_act_list = [F.one_hot(actions[:-1, i], self.action_dim).float() for i in range(self.num_agents)]
        curr_actions_onehot = torch.cat(curr_act_list, dim=-1).unsqueeze(0) # (1, T-1, All_Action_Dim)

        for i in range(2):
            q_val = self.critics[i](states[:, :-1], curr_actions_onehot).squeeze(-1)
            critic_loss = F.mse_loss(q_val, y.detach())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()

        # --- Actor & Alpha 更新 ---
        for i in range(self.num_agents):
            probs, _ = self.actors[i](obs[:, :-1, i], memory.h_actors[0][i])
            log_probs = torch.log(probs + 1e-8)
            
            # 各行動のQ値を計算するために、対象エージェントの行動確率だけを差し替える
            # (簡易化のため、現在の全エージェントの確率分布に基づく期待値を使用)
            # 本来のHASAC/SACの離散版では全アクションのQ値を計算して重み付けます
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Actor Loss: 期待Q値を最大化
            # (Criticの入力に合わせて形状を調整)
            q_val = self.critics[0](states[:, :-1], curr_actions_onehot).squeeze(-1)
            actor_loss = -(torch.exp(self.log_alphas[i]) * entropy + q_val.mean())
            
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()
            
            # Alpha Loss: 探索の強さを調整
            alpha_loss = -(self.log_alphas[i] * (entropy - self.target_entropy).detach()).mean()
            self.alpha_opts[i].zero_grad()
            alpha_loss.backward()
            self.alpha_opts[i].step()

        # ソフトターゲット更新
        for t, c in zip(self.target_critics, self.critics):
            for t_p, c_p in zip(t.parameters(), c.parameters()):
                t_p.data.copy_(t_p.data * (1.0 - self.tau) + c_p.data * self.tau)