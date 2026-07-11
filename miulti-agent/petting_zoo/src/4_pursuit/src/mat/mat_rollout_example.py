"""
既存の PursuitWrapper (添付コード) を変更せずに、MAT_PPO と組み合わせて
ロールアウトを収集するための統合例。

重要なポイント:
    元の MAPPO は各エージェントの行動を独立に(1回のバッチ forward で並列に)
    決定していたので、ロールアウトループは「1体ずつ順に step する」だけで
    自然に機能していました。

    MAT は「エージェント0->7の順に、直前の行動を条件として自己回帰的に
    行動を決める」ため、1サイクルの最初に **8体全員の観測を一度に集めて
    from まとめてデコード** する必要があります。

    幸い、PursuitWrapper.get_global_state() は既に「全エージェントの観測を
    集めて連結する」実装になっているので(かつ、この時点ではまだ誰も
    行動していないため、8体とも同じサイクル開始時点の状態を見ている)、
    このメソッドをそのまま使い、agent_order を possible_agents の順番に
    揃えるだけで済みます。

NOTE: 実行検証はできていません(ネットワーク制限によりPyTorchなし)。
      ロジックの整合性のみ確認したリファレンス実装です。
"""

from __future__ import annotations

import numpy as np

from mat_trainer import MAT_PPO
# 添付いただいた元コードのファイル名を想定 (例: pursuit_wrapper.py)
# from pursuit_wrapper import PursuitWrapper


def collect_rollout(wrapper, mat_ppo: MAT_PPO, n_cycles: int, gamma: float = 0.99,
                     gae_lambda: float = 0.95):
    """
    1エピソード分のロールアウトを収集し、GAEでadvantage/returnsを計算して
    MAT_PPO.update() にそのまま渡せるバッチ(dict)を返す。
    """
    wrapper.reset()

    agent_order = wrapper.possible_agents  # 例: ['pursuer_0', ..., 'pursuer_7']
    num_agents = wrapper.num_agents

    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    for cycle in range(n_cycles):
        # --- 1. サイクル開始時点で、8体全員の観測を一度に取得 ---
        # get_global_state() は possible_agents の順で連結されているので reshape でOK
        joint_obs_flat = wrapper.get_global_state()  # (num_agents * obs_dim,)
        joint_obs = joint_obs_flat.reshape(num_agents, wrapper.obs_dim)  # (8, 236)

        # --- 2. MATで8体分の行動を自己回帰的に一括デコード ---
        actions, log_probs, values = mat_ppo.get_action(joint_obs, greedy=False)
        # actions: (8,), log_probs: (8,), values: (8,)

        # --- 3. agent_order の順に、それぞれの行動を環境に適用 ---
        step_rewards = np.zeros(num_agents, dtype=np.float32)
        step_dones = np.zeros(num_agents, dtype=np.float32)

        for i, agent in enumerate(agent_order):
            if agent not in wrapper.env.agents:
                step_dones[i] = 1.0
                continue
            _, reward, terminated, truncated, _info, _count_capture = wrapper.step(agent, int(actions[i]))
            step_rewards[i] = reward
            step_dones[i] = float(terminated or truncated)

        obs_buf.append(joint_obs)
        act_buf.append(actions)
        logp_buf.append(log_probs)
        val_buf.append(values)
        rew_buf.append(step_rewards)
        done_buf.append(step_dones)

        if len(wrapper.env.agents) == 0:
            break

    obs_arr = np.array(obs_buf, dtype=np.float32)          # (T, 8, obs_dim)
    act_arr = np.array(act_buf, dtype=np.int64)             # (T, 8)
    logp_arr = np.array(logp_buf, dtype=np.float32)         # (T, 8)
    val_arr = np.array(val_buf, dtype=np.float32)           # (T, 8)
    rew_arr = np.array(rew_buf, dtype=np.float32)           # (T, 8)
    done_arr = np.array(done_buf, dtype=np.float32)         # (T, 8)

    advantages, returns = compute_gae(rew_arr, val_arr, done_arr, gamma, gae_lambda)

    batch = {
        "obs": obs_arr,
        "actions": act_arr,
        "log_probs": logp_arr,
        "advantages": advantages,
        "rewards": returns,  # value関数の教師信号として使うため returns を渡す
    }
    return batch


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
    """
    rewards, values, dones: (T, num_agents)
    Returns: advantages (T, num_agents), returns (T, num_agents)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    # エピソード終端後の価値は0として扱う(ブートストラップなし。必要なら最終状態のvalueを渡す形に拡張可)
    next_value = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    print("このファイルは実行例のリファレンスです。")
    print("実際に動かす際は、PursuitWrapper を import し、以下のように呼び出してください:")
    print("""
    from pursuit_wrapper import PursuitWrapper
    from mat_trainer import MAT_PPO

    wrapper = PursuitWrapper(max_cycles=500)
    mat_ppo = MAT_PPO(num_agents=wrapper.num_agents, obs_dim=wrapper.obs_dim, action_dim=wrapper.action_dim)

    for episode in range(1000):
        batch = collect_rollout(wrapper, mat_ppo, n_cycles=500)
        actor_loss, critic_loss, entropy = mat_ppo.update(batch, epochs=3)
        print(f"episode={episode} actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} entropy={entropy:.4f}")
    """)
