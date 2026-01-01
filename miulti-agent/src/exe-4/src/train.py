env = DroneDeliveryEnv()
# HASAC用に次元を計算
obs_dim = 23
state_dim = 46
action_dim = 7
trainer = HASACTrainer(obs_dim, state_dim, action_dim)

for episode in range(1001):
    obs_list = env.reset()
    memory = Memory() # 先ほど修正した Memory クラスを使用
    h_actors = [torch.zeros(1, 1, 128) for _ in range(2)]
    ep_reward = 0
    
    for t in range(env.max_steps):
        obs_tensor = trainer.normalize_obs(obs_list)
        state_tensor = obs_tensor.view(1, -1)
        
        if t == 0:
            memory.h_actors.append([h.clone() for h in h_actors])

        actions = []
        new_h_actors = []
        with torch.no_grad():
            for i in range(2):
                probs, h_a = trainer.actors[i](obs_tensor[i].view(1, 1, -1), h_actors[i])
                dist = Categorical(probs)
                a = dist.sample() # SACは常にサンプリングで探索
                actions.append(a.item())
                new_h_actors.append(h_a)
        
        next_obs_list, rewards, done, _ = env.step(actions)
        
        memory.obs.append(obs_tensor)
        memory.states.append(state_tensor.squeeze())
        memory.actions.append(torch.tensor(actions))
        memory.returns.append(torch.FloatTensor(rewards)) # rewardをreturnsリストに保存
        
        obs_list = next_obs_list
        h_actors = new_h_actors
        ep_reward += sum(rewards)
        if done: break
    
    trainer.train(memory)
    memory.clear()

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {ep_reward:.2f}, Alpha: {torch.exp(trainer.log_alphas[0]).item():.4f}")