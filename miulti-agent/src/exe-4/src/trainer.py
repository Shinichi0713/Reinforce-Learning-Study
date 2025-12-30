import torch.optim as optim

class MAPPOTrainer:
    def __init__(self, obs_dim, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.num_agents = 2
        
        # エージェントごとにActor、共通のCritic
        self.actors = [MAPPO_Actor(obs_dim, action_dim) for _ in range(self.num_agents)]
        self.critic = MAPPO_Critic(state_dim)
        
        self.actor_opts = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def _obs_to_tensor(self, obs_list):
        """dict型の観測をテンソルに変換"""
        tensors = []
        for o in obs_list:
            # 位置(2), 所持(1), 他者位置(2), 荷物(3x4=12) をフラットに
            vec = [o["agent_pos"][0], o["agent_pos"][1], o["carrying"]]
            vec += [o["other_agent"][0], o["other_agent"][1]]
            for p in o["packages"]:
                vec += [p[0][0], p[0][1], p[1][0], p[1][1], 1 if p[2] else 0, 1 if p[3] else 0]
            tensors.append(torch.FloatTensor(vec))
        return torch.stack(tensors)

    def train(self, memory):
        # memoryからデータを展開 (Batch処理)
        states = torch.stack(memory.states)  # (T, State_Dim)
        obs = torch.stack(memory.obs)        # (T, Agents, Obs_Dim)
        actions = torch.stack(memory.actions) # (T, Agents)
        old_log_probs = torch.stack(memory.log_probs) # (T, Agents)
        returns = torch.stack(memory.returns) # (T, Agents)
        
        # Criticの更新
        values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(values, returns.mean(dim=1)) # 全員の平均報酬をターゲットに
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actorの更新 (各エージェント独立)
        advantages = (returns.mean(dim=1) - values.detach())
        
        for i in range(self.num_agents):
            dist = self.actors[i](obs[:, i])
            new_log_probs = dist.log_prob(actions[:, i])
            
            ratio = torch.exp(new_log_probs - old_log_probs[:, i])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

# データ保存用クラス
class Memory:
    def __init__(self):
        self.states, self.obs, self.actions, self.log_probs, self.returns = [], [], [], [], []
    def clear(self):
        self.__init__()

env = DroneDeliveryEnv()
# obs_dim = 2(pos) + 1(carry) + 2(other) + 3pkgs * 6 = 23
# state_dim = obs_dim * num_agents (簡易集中Critic用)
trainer = MAPPOTrainer(obs_dim=23, state_dim=46, action_dim=7)

for episode in range(1000):
    obs_list = env.reset()
    memory = Memory()
    ep_reward = 0
    
    for t in range(env.max_steps):
        obs_t = trainer._obs_to_tensor(obs_list)
        state_t = obs_t.view(-1) # 全エージェントの観測を結合してStateとする
        
        actions, log_probs = [], []
        for i in range(2):
            dist = trainer.actors[i](obs_t[i])
            a = dist.sample()
            actions.append(a.item())
            log_probs.append(dist.log_prob(a))
            
        next_obs_list, rewards, done, _ = env.step(actions)
        
        # メモリに保存 (訓練用に簡略化)
        memory.obs.append(obs_t)
        memory.states.append(state_t)
        memory.actions.append(torch.tensor(actions))
        memory.log_probs.append(torch.stack(log_probs))
        memory.returns.append(torch.FloatTensor(rewards))
        
        obs_list = next_obs_list
        ep_reward += sum(rewards)
        if done: break
    
    trainer.train(memory)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {ep_reward}")
        env.save_gif(agent_model=trainer, filename=f"mappo_ep{episode}.gif")