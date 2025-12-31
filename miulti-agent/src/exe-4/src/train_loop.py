sum_reward = 0.0
interval = 50


env = DroneDeliveryEnv()
trainer = MAPPOTrainer(obs_dim=23, state_dim=46, action_dim=7)

for episode in range(1001):
    obs_list = env.reset()
    memory = Memory()
    
    # --- GRUの初期隠れ状態をリセット ---
    # shape: (num_layers, batch, hidden_dim)
    h_actors = [torch.zeros(1, 1, trainer.hidden_act) for _ in range(2)]
    h_critic = torch.zeros(1, 1, trainer.hidden_crit)
    
    ep_reward = 0
    
    for t in range(env.max_steps):
        # 1. 観測の正規化と整形
        obs_tensor = trainer.normalize_obs(obs_list) # (2, 23)
        state_tensor = obs_tensor.view(1, -1) # (1, 46)
        
        # 学習用に現在のHidden Stateを保存 (最初のステップのみ重要)
        if t == 0:
            memory.h_actors.append([h.clone() for h in h_actors])
            memory.h_critics.append(h_critic.clone())

        # 2. 行動選択
        actions, log_probs = [], []
        new_h_actors = []
        with torch.no_grad():
            for i in range(2):
                # 入力を (batch=1, seq=1, dim) に変換
                dist, h_a = trainer.actors[i](obs_tensor[i].view(1, 1, -1), h_actors[i])
                a = dist.sample()
                actions.append(a.item())
                log_probs.append(dist.log_prob(a))
                new_h_actors.append(h_a)
            
            # Criticの価値計算もHiddenを更新
            _, h_c = trainer.critic(state_tensor.view(1, 1, -1), h_critic)

        # 3. 環境ステップ
        next_obs_list, rewards, done, _ = env.step(actions)
        
        # 4. メモリ保存
        memory.obs.append(obs_tensor)
        memory.states.append(state_tensor.squeeze())
        memory.actions.append(torch.tensor(actions))
        memory.log_probs.append(torch.stack(log_probs).squeeze())
        memory.returns.append(torch.FloatTensor(rewards))
        
        # 状態更新
        obs_list = next_obs_list
        h_actors = new_h_actors
        h_critic = h_c
        ep_reward += sum(rewards)
        
        if done: break
    
    # 5. 学習実行
    trainer.train(memory)
    sum_reward += ep_reward
    if (episode + 1) % interval == 0:
        print(f"{episode} : {sum_reward}")
        sum_reward = 0.0
    if episode % 50 == 0:
        print(f"Episode {episode} | Total Reward: {ep_reward:.2f}")