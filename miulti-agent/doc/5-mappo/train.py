env = DroneDeliveryEnv()
trainer = MAPPOTrainer_V2(obs_dim=23, action_dim=7)

for episode in range(1001):
    obs_list = env.reset()
    memory = MAPPOMemory()
    
    h_actors = [torch.zeros(1, 1, 128) for _ in range(2)]
    h_critic = torch.zeros(1, 1, 256)
    
    # 最初のHidden Stateを保存
    memory.h_actors.append([h.clone() for h in h_actors])
    memory.h_critics.append(h_critic.clone())
    
    for t in range(env.max_steps):
        # 1. 正規化観測を取得
        obs_tensor = trainer.normalize_obs(obs_list) # (2, 23)
        # 2. 集中Critic用のGlobal State作成 (全員分結合)
        global_state = obs_tensor.view(-1) # (46,)
        
        # 3. 行動決定
        actions, log_probs, next_h_actors = [], [], []
        for i in range(2):
            agent_id = torch.zeros(2); agent_id[i] = 1.0
            inp = torch.cat([obs_tensor[i], agent_id], dim=-1).view(1, 1, -1)
            
            with torch.no_grad():
                dist, h_a = trainer.actor(inp, h_actors[i])
                a = dist.sample()
                actions.append(a.item())
                log_probs.append(dist.log_prob(a))
                next_h_actors.append(h_a)
        
        # 4. 環境ステップ
        next_obs_list, rewards, done, _ = env.step(actions)
        
        # 5. 保存
        memory.store(obs_tensor, global_state, torch.tensor(actions), 
                     torch.stack(log_probs).squeeze(), torch.FloatTensor(rewards), done)
        
        obs_list = next_obs_list
        h_actors = next_h_actors
        if done: break
        
    # 6. 学習
    trainer.train(memory)