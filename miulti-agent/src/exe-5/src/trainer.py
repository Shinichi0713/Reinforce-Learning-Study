import gym
import numpy as np
import torch

def train_qmix():
    # 環境設定
    env = CooperativeNavigationEnv(size=5)  # size=5, num_agents=2 固定
    num_agents = env.num_agents
    obs_dim = len(env._get_obs()[0])  # 観測の次元（_get_obsで確認）
    action_dim = 5  # 行動空間（0:停止, 1:上, 2:下, 3:左, 4:右）
    state_dim = obs_dim * num_agents  # グローバル状態の次元（全観測の結合）

    # ハイパーパラメータ
    num_episodes = 2000
    max_steps = env.max_steps
    batch_size = 32
    gamma = 0.95
    lr = 1e-3

    # Trainer と Memory の初期化
    trainer = QMIXTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        state_dim=state_dim,
        gamma=gamma,
        lr=lr
    )
    memory = QMIXMemory()

    # 学習ループ
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step = 0

        # エピソード内ループ
        while step < max_steps:
            # 観測をテンソルに変換
            obs_tensor = trainer.normalize_obs(obs)  # (num_agents, obs_dim)

            # 行動選択（学習時は training=True）
            actions = []
            for i in range(num_agents):
                action = trainer.select_action(obs_tensor, i, training=True)
                actions.append(action)

            # 環境ステップ
            next_obs, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards)

            # グローバル状態の構築（全エージェントの観測を結合）
            state = np.concatenate(obs)
            next_state = np.concatenate(next_obs)

            # メモリに保存
            memory.store(
                obs=obs_tensor,
                actions=torch.tensor(actions, dtype=torch.long),
                rewards=torch.tensor(rewards, dtype=torch.float),
                next_obs=trainer.normalize_obs(next_obs),
                done=done,
                state=torch.FloatTensor(state),
                next_state=torch.FloatTensor(next_state)
            )

            # バッチサイズに達したら学習
            if len(memory.obs) >= batch_size:
                loss = trainer.train(memory)
                memory.clear()  # 1バッチ分学習したらクリア（オンポリシック風）

            obs = next_obs
            step += 1

            if done:
                break

        # エピソード終了時に温度を減衰
        trainer.update_temperature()

        # ログ出力
        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {episode_reward:.2f}, "
                  f"Temperature: {trainer.temperature:.3f}")

    # 学習終了後、評価用の行動選択（training=False）
    def evaluate(num_eval_episodes=10):
        for eval_ep in range(num_eval_episodes):
            obs = env.reset()
            eval_reward = 0
            step = 0
            while step < max_steps:
                obs_tensor = trainer.normalize_obs(obs)
                actions = []
                for i in range(num_agents):
                    action = trainer.select_action(obs_tensor, i, training=False)
                    actions.append(action)
                next_obs, rewards, done, _ = env.step(actions)
                eval_reward += sum(rewards)
                obs = next_obs
                step += 1
                if done:
                    break
            print(f"Eval Episode {eval_ep}, Reward: {eval_reward:.2f}")

    evaluate()

    # GIF保存（学習済みモデルで）
    env.save_gif(trainer=trainer, filename="coop_nav_qmix.gif")

if __name__ == "__main__":
    train_qmix()