"""
MAT (Multi-Agent Transformer) 版の PPO トレーナー。

元コードの `MAPPO` クラスとほぼ同じ呼び出しインターフェース
(`get_action`, `update`, `load_checkpoint` 等) を保つように設計しているので、
既存の学習ループ・ロールアウト収集コードへの変更を最小限にできます。

主な違い:
  - Actor と Critic が1つの MATActorCritic に統合されている
    (Encoderの出力を両方が共有するため)
  - get_action が「8体分の行動をまとめて」自己回帰的にデコードする
    (元のMAPPOは8体を独立に、1回のバッチ forward で並列に決定していた)
  - update 内の ratio 計算は、教師強制で再計算した対数確率を使う
    (デコーダの構造上、各エージェントの対数確率は前のエージェントの実際の行動に
    条件づけられているため、元のMAPPOのように各エージェントを完全独立には
    扱えないが、損失関数の形自体はPPOクリップのまま変わらない)

NOTE: この環境ではネットワーク制限によりPyTorchの実行検証ができないため、
      構文チェックのみ実施しています。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mat_actor_critic import MATActorCritic


class MAT_PPO:
    def __init__(self, num_agents=8, obs_dim=236, action_dim=5,
                 d_model=64, nhead=4, spatial_layers=2, enc_layers=2, dec_layers=2,
                 lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 device=torch.device("cpu")):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        self.model = MATActorCritic(
            d_model=d_model, nhead=nhead,
            spatial_layers=spatial_layers, enc_layers=enc_layers, dec_layers=dec_layers,
            act_dim=action_dim, num_agents=num_agents,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, joint_obs: np.ndarray, greedy: bool = False):
        """
        joint_obs: (num_agents, obs_dim) の numpy 配列。
                   possible_agents と同じ順番(pursuer_0, pursuer_1, ..., pursuer_7)で
                   並んでいる必要がある(デコーダの自己回帰順序と一致させるため)。

        Returns: actions (num_agents,), log_probs (num_agents,), values (num_agents,)
                 いずれも numpy 配列
        """
        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, log_probs, values = self.model.act(obs_t, greedy=greedy)
        return (
            actions.squeeze(0).cpu().numpy(),
            log_probs.squeeze(0).cpu().numpy(),
            values.squeeze(0).cpu().numpy(),
        )

    def update(self, batch: dict, epochs: int = 3):
        """
        batch は元のMAPPOと同じキーを想定:
            'obs':        (B, num_agents, obs_dim)
            'actions':    (B, num_agents)
            'log_probs':  (B, num_agents)  ロールアウト時点(pi_old)の対数確率
            'advantages': (B, num_agents)
            'rewards':    (B, num_agents)  GAEのリターン(価値関数の教師信号)
        """
        if batch is None:
            return 0.0, 0.0, 0.0

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(epochs):
            new_log_probs, entropy, values = self.model.forward_train(obs, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            critic_loss = nn.SmoothL1Loss()(values, returns) * self.value_coef

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())

        return float(np.mean(actor_losses)), float(np.mean(critic_losses)), float(np.mean(entropies))

    def save_checkpoint(self, path: str, episode: int):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
        }, path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode = checkpoint["episode"]
        print(f"チェックポイントを読み込み: {checkpoint_path} (episode: {episode})")
        return episode
