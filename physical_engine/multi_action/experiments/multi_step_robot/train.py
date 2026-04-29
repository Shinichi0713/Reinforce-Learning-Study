
def train_sac(env, agent, buffer, max_episodes=1000, max_steps=200, batch_size=256, start_steps=10000):
    episode_rewards = []

    total_steps = 0
    for episode in range(max_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 探索ステップまではランダム行動
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)

            next_obs, reward, done, truncated, info = env.step(action)
            buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            total_steps += 1

            # バッチサイズ分たまったら学習
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                q_loss, policy_loss, alpha = agent.update(batch, batch_size)
                if total_steps % 1000 == 0:
                    print(f"Step {total_steps}, Q Loss: {q_loss:.4f}, Policy Loss: {policy_loss:.4f}, Alpha: {alpha:.4f}")

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {total_steps}")

        # 一定間隔でモデルを保存（任意）
        if episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"policy_{episode}.pth")

    return episode_rewards


# 環境の準備（前回の RobotCarryEnv を想定）
env = RobotCarryEnv(max_steps=200, world_size=10.0)

# 観測次元・行動次元を環境から取得（または手動で指定）
obs_dim = env.observation_space.shape[0]  # 例: 19
act_dim = env.action_space.shape[0]      # 例: 2

# エージェントとバッファの初期化
agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256, lr=3e-4)
buffer = ReplayBuffer(capacity=100000)

# 学習実行
rewards = train_sac(
    env=env,
    agent=agent,
    buffer=buffer,
    max_episodes=1000,
    max_steps=200,
    batch_size=256,
    start_steps=10000
)

env.close()