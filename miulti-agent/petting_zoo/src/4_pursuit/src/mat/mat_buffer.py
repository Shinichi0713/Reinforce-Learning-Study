"""
MAT_PPO 用のロールアウトバッファ。

役割:
  1. 1回のロールアウト(n_steps分)で集めた (obs, actions, log_probs, values,
     rewards, dones) を時系列順に蓄積する。
  2. ロールアウト終了後、GAE (Generalized Advantage Estimation) で
     advantages / returns を計算する。
  3. 学習時に epochs x シャッフルされたミニバッチ という形で
     MAT_PPO.update() に渡せる形式のバッチを順次生成する。

設計上のポイント:
  - MATActorCritic は時刻方向に再帰構造を持たない(自己回帰はエージェント方向のみ)
    ため、時刻(タイムステップ)をまたいだシャッフルは安全に行えます。
    ミニバッチはタイムステップ次元でシャッフル・分割し、各サンプルは
    (num_agents, obs_dim) の1ステップ分の同時デシジョンをそのまま保持します。
  - dones は「エピソード全体が終了したか」を表すスカラー的な意味合いとして、
    全エージェントに同じ値を入れる想定です(Pursuitはチーム全員が同時に
    エピソード終了するため)。

NOTE: 実行環境の制約(ネットワーク遮断)によりPyTorch自体は動かせないため、
      本ファイルはNumPyのみに依存しており、構文チェックのみ実施しています。
"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, buffer_size: int, num_agents: int, obs_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)

        self.ptr = 0

    def add(self, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray,
            values: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        1タイムステップ分(8エージェント同時決定)のデータを追加する。

        obs:       (num_agents, obs_dim)
        actions:   (num_agents,)
        log_probs: (num_agents,)
        values:    (num_agents,)
        rewards:   (num_agents,)
        dones:     (num_agents,)
        """
        if self.ptr >= self.buffer_size:
            raise RuntimeError("バッファが満杯です。reset() してから追加してください。")

        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        GAE (Generalized Advantage Estimation) で advantages / returns を計算する。

        last_values: (num_agents,) バッファ末尾の次の状態でのValue推定
                     (ロールアウトが打ち切られた場合のブートストラップ用)
        last_dones:  (num_agents,) バッファ末尾の次の状態がエピソード終了直後かどうか
        """
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        n = self.ptr

        for step in reversed(range(n)):
            if step == n - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int | None = None, shuffle: bool = True):
        """
        学習用のミニバッチを順に生成するジェネレータ。
        1エポック分(バッファ全体を1回ずつ使い切る)のミニバッチを生成する。
        エポックを複数回まわしたい場合は、学習ループ側でこのメソッドを
        エポック数だけ繰り返し呼び出してください(呼ぶたびに再シャッフルされます)。
        """
        n = self.ptr
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        if batch_size is None:
            batch_size = n

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "rewards": self.returns[batch_idx],  # Critic の教師信号として returns を使う
            }


if __name__ == "__main__":
    # 簡易動作確認 (NumPyのみ、ダミーデータでGAE計算とミニバッチ生成をテスト)
    rng = np.random.default_rng(0)
    T, N, D = 20, 8, 236

    buf = RolloutBuffer(buffer_size=T, num_agents=N, obs_dim=D, gamma=0.99, gae_lambda=0.95)

    for t in range(T):
        obs = rng.normal(size=(N, D)).astype(np.float32)
        actions = rng.integers(0, 5, size=(N,))
        log_probs = rng.normal(size=(N,)).astype(np.float32)
        values = rng.normal(size=(N,)).astype(np.float32)
        rewards = rng.normal(size=(N,)).astype(np.float32)
        dones = np.zeros(N, dtype=np.float32)
        buf.add(obs, actions, log_probs, values, rewards, dones)

    last_values = rng.normal(size=(N,)).astype(np.float32)
    last_dones = np.zeros(N, dtype=np.float32)
    buf.compute_returns_and_advantages(last_values, last_dones)

    print(f"advantages shape: {buf.advantages.shape}, returns shape: {buf.returns.shape}")
    n_batches = 0
    for mb in buf.get_batches(batch_size=8, shuffle=True):
        n_batches += 1
        assert mb["obs"].shape[1:] == (N, D)
    print(f"生成されたミニバッチ数: {n_batches} (T={T}, batch_size=8 -> ceil(20/8)=3 のはず)")
    print("OK: RolloutBuffer の基本動作を確認しました。")
