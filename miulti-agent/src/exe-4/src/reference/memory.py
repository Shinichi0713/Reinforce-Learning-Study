def train(self, memory):
        if len(memory.states) == 0: return

        # stackしてテンソル化
        states = torch.stack(memory.states)
        obs = torch.stack(memory.obs)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        returns = torch.stack(memory.returns)
        
        # --- 累積報酬（Returns）の計算（簡易版：将来の報酬を足し合わせる） ---
        # 単にその場でもらった報酬(returns)を使うだけだと学習が不安定なので、
        # 本来はここで報酬の累積和を計算するのが望ましいです。
        
        # Critic更新
        values = self.critic(states).squeeze()
        # ターゲットは勾配計算から外す（detach）
        critic_loss = F.mse_loss(values, returns.mean(dim=1).detach())
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor更新
        # アドバンテージ計算時もvaluesをdetachする
        advantages = (returns.mean(dim=1) - values.detach())
        
        for i in range(self.num_agents):
            dist = self.actors[i](obs[:, i])
            new_log_probs = dist.log_prob(actions[:, i])
            
            # PPOの公式: ratio = new / old
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            # 最大化したいのでマイナスを付けて最小化問題にする
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            
            self.actor_opts[i].zero_grad()
            actor_loss.backward() # これでこのエピソードのグラフが解放される
            self.actor_opts[i].step()
            
        # 最後にメモリを空にする
        memory.clear()