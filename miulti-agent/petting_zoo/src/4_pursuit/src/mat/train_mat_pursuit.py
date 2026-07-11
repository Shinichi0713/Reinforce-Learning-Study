"""
MAT_PPO + PursuitWrapper + RolloutBuffer を組み合わせた学習ループ。

流れ:
  1. n_steps 分のロールアウトを収集し RolloutBuffer に貯める
     (エピソードがn_steps未満で終了したら wrapper.reset() して継続収集する)
  2. ロールアウト終了時点の状態でValueを推定し、GAEでadvantages/returnsを計算
  3. epochs 回、シャッフルしたミニバッチ単位で MAT_PPO.update() を呼び、
     勾配更新する
  4. これを total_updates 回繰り返す

NOTE: 実行環境の制約(ネットワーク遮断)によりPyTorchが使えないため、
      このファイルは構文チェックのみ実施済みで、実行検証はできていません。
      import 元のファイル名 (pursuit_wrapper.py) はご自身の環境のファイル名に
      合わせて変更してください。
"""

from __future__ import annotations

import os
import numpy as np
import torch

from mat_trainer import MAT_PPO
from mat_buffer import RolloutBuffer

# 添付いただいた元コードのファイル名を想定しています。
# 実際のファイル名に合わせて変更してください (例: from pursuit_env import PursuitWrapper)
from pursuit_wrapper import PursuitWrapper


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

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    wrapper.reset()

    episode_reward = np.zeros(num_agents, dtype=np.float32)
    episode_captures = 0
    episode_count = 0

    for update_idx in range(1, total_updates + 1):
        buffer.reset()

        # =========================================================
        # 1. ロールアウト収集
        # =========================================================
        for _step in range(n_steps):
            joint_obs_flat = wrapper.get_global_state()
            joint_obs = joint_obs_flat.reshape(num_agents, obs_dim)

            # MATで8体分の行動を自己回帰的に一括デコード (サンプリング, 学習モード)
            actions, log_probs, values = mat_ppo.get_action(joint_obs, greedy=False)

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
        _, _, last_values = mat_ppo.get_action(joint_obs, greedy=True)
        last_dones = np.zeros(num_agents, dtype=np.float32)  # バッファ終端は通常「継続中」

        buffer.compute_returns_and_advantages(last_values, last_dones)

        # =========================================================
        # 3. PPO更新: epochs回、シャッフルしたミニバッチで学習
        # =========================================================
        actor_losses, critic_losses, entropies = [], [], []
        for _epoch in range(epochs):
            for minibatch in buffer.get_batches(batch_size=batch_size, shuffle=True):
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
    import argparse

    parser = argparse.ArgumentParser(description="MAT + Pursuit 学習スクリプト")
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        total_updates=args.total_updates,
        n_steps=args.n_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
