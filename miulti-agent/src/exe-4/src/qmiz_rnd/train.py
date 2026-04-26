env = DroneDeliveryEnv(grid_size=10, num_agents=2, num_packages=3, max_steps=200)

# 状態次元の計算
state_dim_per_agent = (
    2  # agent_pos
    + 1  # carrying
    + 2  # other_agent_pos
    + 6 * env.num_packages  # packages情報
)
action_dim = 7  # 0-6

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = QMIXRNDAgent(env, state_dim_per_agent, action_dim, device=device)


episodes = 1000
update_interval = 4  # 4ステップごとに更新など

for ep in range(episodes):
    obs = env.reset()
    total_rewards = [0, 0]
    step = 0

    while True:
        # 行動選択
        actions = agent.act(obs, explore=True)

        # 環境ステップ
        next_obs, rewards, done, _ = env.step(actions)

        # RND による内部報酬を計算
        intrinsic_rewards = agent.compute_intrinsic_reward(obs)
        total_rewards = [r + ext + intr for r, ext, intr in zip(total_rewards, rewards, intrinsic_rewards)]

        # 外部報酬 + 内部報酬で学習
        combined_rewards = [ext + intr for ext, intr in zip(rewards, intrinsic_rewards)]

        # 遷移を保存
        agent.store_transition(obs, actions, combined_rewards, next_obs, done)

        # RND ネットワークの更新（頻度は調整可能）
        agent.update_rnd(obs)

        # QMIX の更新
        if step % update_interval == 0:
            agent.update_qmix()

        obs = next_obs
        step += 1

        if done:
            break

    # ターゲットネットワークの更新
    if ep % TARGET_UPDATE == 0:
        agent.update_targets()

    if ep % 10 == 0:
        print(f"Episode {ep}, Rewards: {total_rewards}, Eps: {agent.eps:.3f}")