import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAPPOAgent(nn.Module):
    def __init__(self, action_space_n=18):
        super().__init__()

        # --- 共通のCNNバックボーン (特徴抽出器) ---
        # ActorもCriticも同じ構造のCNNを使用しますが、重みは別々に管理します
        def make_cnn(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU()
            )

        # Actor: 自分の4フレーム分を見る (in=4)
        self.actor_encoder = make_cnn(in_channels=4)
        self.action_head = nn.Linear(512, action_space_n)

        # Centralized Critic: 自分と相手の計8フレームを見る (in=8)
        self.critic_encoder = make_cnn(in_channels=8)
        # 1P用と2P用の価値をそれぞれ出力するヘッド
        self.value_head_1p = nn.Linear(512, 1)
        self.value_head_2p = nn.Linear(512, 1)

    def get_action(self, obs, action=None):
        """
        Actor: 行動と対数確率、エントロピーを返す
        obs: (batch, 4, 84, 84)
        """
        features = self.actor_encoder(obs)
        logits = self.action_head(features)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, joint_obs):
        """
        Centralized Critic: 神の視点での評価値を返す
        joint_obs: (batch, 8, 84, 84)
        """
        features = self.critic_encoder(joint_obs)
        v1 = self.value_head_1p(features)
        v2 = self.value_head_2p(features)
        return v1, v2


class MAPPORolloutBuffer:
    def __init__(self, buffer_size, obs_shape, joint_shape, device):
        # ここで self.buffer_size として保存する必要があります
        self.buffer_size = buffer_size 
        self.device = device
        
        # メモリ節約のための uint8 設定
        self.obs = torch.zeros((buffer_size, 2, *obs_shape), dtype=torch.uint8, device=device)
        self.joint_states = torch.zeros((buffer_size, *joint_shape), dtype=torch.uint8, device=device)
        # その他の報酬やアドバンテージは float32 のまま
        self.actions = torch.zeros((buffer_size, 2), dtype=torch.long, device=device)

        self.log_probs = torch.zeros((buffer_size, 2), device=device)
        self.rewards = torch.zeros((buffer_size, 2), device=device)
        self.values = torch.zeros((buffer_size, 2), device=device)
        self.dones = torch.zeros((buffer_size, 2), device=device)

        self.ptr = 0

    def insert(self, obs_1p, obs_2p, joint_state, actions, log_probs, rewards, values, dones):
        """1ステップ分のデータを格納"""
        self.obs[self.ptr, 0] = obs_1p
        self.obs[self.ptr, 1] = obs_2p
        self.joint_states[self.ptr] = joint_state

        # actions, log_probs 等は [1Pの値, 2Pの値] のリストや配列を想定
        self.actions[self.ptr] = torch.tensor(actions, device=self.device)
        self.log_probs[self.ptr] = torch.tensor(log_probs, device=self.device)
        self.rewards[self.ptr] = torch.tensor(rewards, device=self.device)
        self.values[self.ptr] = torch.tensor(values, device=self.device)
        self.dones[self.ptr] = torch.tensor(dones, device=self.device)

        self.ptr = (self.ptr + 1) % self.buffer_size

    def get_batches(self, batch_size):
        """学習用にデータをシャッフルしてバッチを生成するイテレータ"""
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # 各データのバッチを辞書で返す
            yield {
                "obs": self.obs[batch_idx],
                "joint_states": self.joint_states[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "rewards": self.rewards[batch_idx],
                "values": self.values[batch_idx],
                "dones": self.dones[batch_idx]
            }

    def clear(self):
        """更新後にバッファをリセット"""
        self.ptr = 0

    def get_batches(self, batch_size):
        """学習用にデータをシャッフルしてバッチを生成するイテレータ"""
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # 修正箇所: 'advantages' と 'returns' を辞書に追加
            yield {
                "obs": self.obs[batch_idx],
                "joint_states": self.joint_states[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "rewards": self.rewards[batch_idx],
                "values": self.values[batch_idx],
                "dones": self.dones[batch_idx],
                "advantages": self.advantages[batch_idx], # 追加
                "returns": self.returns[batch_idx]       # 追加
            }

    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        """
        GAEと報酬の期待値(Returns)を計算する
        last_values: 最後のステップの次の状態に対する価値予測 (2, )
        """
        # アドバンテージとリターンを格納する領域を確保
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)

        # 各エージェントごとに計算
        for agent_id in range(2):
            gae = 0
            # 最後の価値予測値を初期値にする
            next_value = last_values[agent_id]

            # バッファを後ろから順に走査 (T-1, T-2, ..., 0)
            for step in reversed(range(self.buffer_size)):
                # TD誤差 (delta) = 報酬 + γ * 次の価値 * (1-done) - 現在の価値
                # ※ doneが1の時は、次のステップの価値を無視する
                delta = self.rewards[step, agent_id] + \
                        gamma * next_value * (1 - self.dones[step, agent_id]) - \
                        self.values[step, agent_id]

                # GAE = delta + γ * λ * (1-done) * 前のステップのgae
                gae = delta + gamma * gae_lambda * (1 - self.dones[step, agent_id]) * gae

                self.advantages[step, agent_id] = gae
                self.returns[step, agent_id] = self.advantages[step, agent_id] + self.values[step, agent_id]

                # 次のループのために今の価値を保持
                next_value = self.values[step, agent_id]

        # アドバンテージの標準化 (学習の安定化のため)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MAPPOAgent().to(device)

    # 1. データの準備 (前回の preprocess_joint_obs を使用)
    o1, o2, joint_s = preprocess_joint_obs(obs_dict, device)

    # 2. Actorによる行動決定 (重みを共有して2人分計算)
    a1, log_p1, _ = agent.get_action(o1.unsqueeze(0))
    a2, log_p2, _ = agent.get_action(o2.unsqueeze(0))

    # ボクシングの18アクション（公式ドキュメント準拠）
    ACTION_MEANING = [
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", 
        "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
        "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
        "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
    ]

    # --- 実行例 ---
    o1, o2, _ = preprocess_joint_obs(obs_dict)
    visualize_action_probs(agent, o1, agent_name="1P (White)")
