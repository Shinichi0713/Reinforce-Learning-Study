def train_maven():
    # 環境設定
    env = CooperativeNavigationEnv(size=5)
    num_agents = env.num_agents
    obs_dim = len(env._get_obs()[0])
    action_dim = 5
    state_dim = obs_dim * num_agents
    z_dim = 4

    # ハイパーパラメータ
    num_episodes = 2000
    max_steps = env.max_steps
    batch_size = 32
    gamma = 0.95
    lr = 1e-3
    beta = 0.01

    # Trainer と Memory の初期化
    trainer = MAVENTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        state_dim=state_dim,
        z_dim=z_dim,
        gamma=gamma,
        lr=lr,
        beta=beta,
        batch_size=batch_size
    )
    memory = MAVENMemory()

    # 損失の初期化（ログ出力用）
    loss_qmix = 0.0
    loss_vae = 0.0

    for ep in range(num_episodes):
        obs = env.reset()
        state = np.concatenate(obs)
        z = trainer.sample_z(torch.FloatTensor(state).to(device))
        episode_reward = 0
        step = 0

        while step < max_steps:
            obs_tensor = trainer.normalize_obs(obs)
            actions = []
            for i in range(num_agents):
                z_expanded = z.unsqueeze(0)  # (1, z_dim)
                action = trainer.select_action(obs_tensor, i, z_expanded, training=True)
                actions.append(action)

            next_obs, rewards, done, _ = env.step(actions)
            next_state = np.concatenate(next_obs)
            episode_reward += sum(rewards)

            # メモリに保存する z は (T, z_dim) の形状に統一
            z_batch = z.unsqueeze(0).repeat(len(obs_tensor), 1)  # (T, z_dim)

            memory.store(
                obs=obs_tensor,
                actions=torch.tensor(actions, dtype=torch.long).to(device),
                rewards=torch.tensor(rewards, dtype=torch.float).to(device),
                next_obs=trainer.normalize_obs(next_obs),
                done=done,
                state=torch.FloatTensor(state).to(device),
                next_state=torch.FloatTensor(next_state).to(device),
                z=z_batch
            )

            if len(memory.obs) >= batch_size:
                loss_qmix, loss_vae = trainer.train(memory)
                memory.clear()

            obs = next_obs
            state = next_state
            step += 1

            if done:
                break

        # 温度パラメータの更新
        trainer.update_temperature()

        # ログ出力
        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {episode_reward:.2f}, "
                  f"Loss QMIX: {loss_qmix:.4f}, Loss VAE: {loss_vae:.4f}")

    # 評価用（z を固定して argmax で行動選択）
    def evaluate(num_eval_episodes=10):
        for eval_ep in range(num_eval_episodes):
            obs = env.reset()
            state = np.concatenate(obs)
            z = trainer.sample_z(torch.FloatTensor(state).to(device))
            eval_reward = 0
            step = 0
            while step < max_steps:
                obs_tensor = trainer.normalize_obs(obs)
                actions = []
                for i in range(num_agents):
                    z_expanded = z.unsqueeze(0)
                    action = trainer.select_action(obs_tensor, i, z_expanded, training=False)
                    actions.append(action)
                next_obs, rewards, done, _ = env.step(actions)
                eval_reward += sum(rewards)
                obs = next_obs
                step += 1
                if done:
                    break
            print(f"Eval Episode {eval_ep}, Reward: {eval_reward:.2f}")

    evaluate()

    # GIF保存（z を固定して可視化）
    env.save_gif(trainer=trainer, filename="coop_nav_maven.gif")

if __name__ == "__main__":
    train_maven()