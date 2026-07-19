import os
import glob
import numpy as np
import torch


def train(
    total_updates: int = 1000,
    n_steps: int = 500,            # 1回のロールアウトで集めるステップ数(バッファサイズ)
    epochs: int = 3,               # 1回のロールアウトあたりのPPOエポック数
    batch_size: int = 128,         # ミニバッチサイズ(タイムステップ方向)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 1e-4,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    checkpoint_every: int = 50,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 1,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # --- 環境・エージェント・バッファの初期化 ---
    wrapper = PursuitWrapper(max_cycles=n_steps)
    num_agents = wrapper.num_agents
    obs_dim = wrapper.obs_dim
    action_dim = wrapper.action_dim
    agent_order = wrapper.possible_agents  # デコーダの自己回帰順序と一致させる

    mat_ppo = MAT_PPO(
        num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon, value_coef=value_coef, entropy_coef=entropy_coef,
        device=device,
    )

    buffer = RolloutBuffer(n_steps, num_agents, obs_dim, gamma=gamma, gae_lambda=gae_lambda)

    # --- チェックポイントのロード処理 ---
    start_update_idx = 1
    if os.path.exists(checkpoint_dir):
        # ディレクトリ内の mat_pursuit_update*.pt を検索
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "mat_pursuit_update*.pt"))
        if ckpt_files:
            # ファイル名からupdateの数値を抽出して最新のものを特定
            # 例: "checkpoints/mat_pursuit_update100.pt" -> 100
            try:
                ckpt_files.sort(key=lambda x: int(os.path.basename(x).split("update")[-1].split(".pt")[0]))
                latest_ckpt = ckpt_files[-1]

                # 保存時のupdate_idxを取得して再開位置を設定
                saved_update_idx = int(os.path.basename(latest_ckpt).split("update")[-1].split(".pt")[0])

                # MAT_PPOのロード関数を呼び出し (引数は実装に合わせて調整してください)
                mat_ppo.load_checkpoint(latest_ckpt)

                start_update_idx = saved_update_idx + 1
                print(f"Loaded checkpoint from: {latest_ckpt} (Resuming from update {start_update_idx})")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")
        else:
            print("No checkpoint found in directory. Starting from scratch.")
    else:
        print("Checkpoint directory does not exist. Starting from scratch.")

    wrapper.reset()

    episode_reward = np.zeros(num_agents, dtype=np.float32)
    episode_captures = 0
    episode_count = 0

    # ロードされたインデックスからループを開始
    for update_idx in range(start_update_idx, total_updates + 1):
        buffer.reset()

        # =========================================================
        # 1. ロールアウト収集
        # =========================================================
        for _step in range(n_steps):
            joint_obs_flat = wrapper.get_global_state()
            joint_obs = joint_obs_flat.reshape(num_agents, obs_dim)

            # MATで8体分の行動を自己回帰的に一括デコード (サンプリング, 学習モード)
            actions, log_probs, values = mat_ppo.get_action(joint_obs, greedy=False)
            # print(actions)

            step_rewards = np.zeros(num_agents, dtype=np.float32)
            step_dones = np.zeros(num_agents, dtype=np.float32)

            for i, agent in enumerate(agent_order):
                if agent not in wrapper.env.agents:
                    step_dones[i] = 1.0
                    continue
                _, reward, terminated, truncated, _info, count_capture = wrapper.step(agent, int(actions[i]))
                step_rewards[i] = reward
                step_dones[i] = float(terminated or truncated)
                episode_captures += count_capture

            buffer.add(joint_obs, actions, log_probs, values, step_rewards, step_dones)
            episode_reward += step_rewards

            # エピソード終了判定 (Pursuitはチーム全員が同時に終了する)
            if len(wrapper.env.agents) == 0:
                episode_count += 1
                if episode_count % log_every == 0:
                    print(
                        f"[episode {episode_count}] "
                        f"reward_sum={episode_reward.sum():.2f} "
                        f"mean_reward={episode_reward.mean():.3f} "
                        f"captures={episode_captures}"
                    )
                wrapper.reset()
                episode_reward = np.zeros(num_agents, dtype=np.float32)
                episode_captures = 0

        # =========================================================
        # 2. GAE計算 (バッファ末尾の次状態をブートストラップに使用)
        # =========================================================
        joint_obs_flat = wrapper.get_global_state()
        joint_obs = joint_obs_flat.reshape(num_agents, obs_dim)
        # greedy=True: ブートストラップ用のValue推定なので行動のサンプリングは不要だが
        # get_action の実装上、Valueは行動デコードと同時に得られるためこの呼び方でよい
        _, _, last_values = mat_ppo.get_action(joint_obs, greedy=False)
        last_dones = np.zeros(num_agents, dtype=np.float32)  # バッファ終端は通常「継続中」

        buffer.compute_returns_and_advantages(last_values, last_dones)

        # =========================================================
        # 3. PPO更新: epochs回、シャッフルしたミニバッチで学習
        # =========================================================
        actor_losses, critic_losses, entropies = [], [], []
        for _epoch in range(epochs):
            for minibatch in buffer.get_batches(batch_size=batch_size, shuffle=False):
                # epochs=1: 1回のミニバッチにつき1回の勾配更新
                a_loss, c_loss, ent = mat_ppo.update(minibatch, epochs=1)
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropies.append(ent)

        if update_idx % log_every == 0:
            print(
                f"update={update_idx}/{total_updates} "
                f"actor_loss={np.mean(actor_losses):.4f} "
                f"critic_loss={np.mean(critic_losses):.4f} "
                f"entropy={np.mean(entropies):.4f}"
            )

        # =========================================================
        # 4. チェックポイント保存
        # =========================================================
        if update_idx % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"mat_pursuit_update{update_idx}.pt")
            mat_ppo.save_checkpoint(path, update_idx)
            print(f"checkpoint saved: {path}")

    return mat_ppo


if __name__ == "__main__":
    # 設定パラメータを直接変数にハードコード
    TOTAL_UPDATES = 1000
    N_STEPS = 500
    EPOCHS = 3
    BATCH_SIZE = 128
    LR = 1e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    CHECKPOINT_EVERY = 50
    CHECKPOINT_DIR = CHECKPOINT_DIR  # 前回の記述エラー（自己代入）も修正
    SEED = 24

    # 関数呼び出しにハードコードした変数を適用
    train(
        total_updates=TOTAL_UPDATES,
        n_steps=N_STEPS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        entropy_coef=ENTROPY_COEF,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_dir=CHECKPOINT_DIR,
        seed=SEED,
    )