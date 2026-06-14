import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape=(7,7,3), channels=(16, 32)):
        super().__init__()
        c_in = obs_shape[-1]  # 3
        c1, c2 = channels     # 16, 32

        self.cnn = nn.Sequential(
            nn.Conv2d(c_in, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # CNN出力の次元を計算（7*7*32 = 1568）
        self.output_dim = 7 * 7 * c2

    def forward(self, obs):
        """
        Args:
            obs: (batch, 147) のフラット観測テンソル
        Returns:
            features: (batch, cnn_output_dim) の特徴ベクトル
        """
        # (batch, 147) -> (batch, 7, 7, 3)
        obs = obs.view(-1, 7, 7, 3)
        # (batch, 7, 7, 3) -> (batch, 3, 7, 7)
        obs = obs.permute(0, 3, 1, 2)
        features = self.cnn(obs)
        return features


# Actor（ポリシー）
class Actor(nn.Module):
    def __init__(self, cnn_encoder, act_dim=5, hidden_size=64):
        super().__init__()
        self.cnn = cnn_encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn.output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs):
        features = self.cnn(obs)
        logits = self.mlp(features)
        dist = Categorical(logits=logits)
        return dist


class Critic(nn.Module):
    def __init__(self, cnn_encoder, num_agents=8, hidden_size=64):
        super().__init__()
        self.cnn = cnn_encoder
        # グローバル状態の次元: num_agents * obs_dim（147*8=1176）
        # CNNエンコーダの出力次元（1568）とは別物なので注意
        state_dim = num_agents * 147  # 147は観測次元（7*7*3）
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, global_state):
        value = self.mlp(global_state)
        return value



class MAPPO:
    """MAPPOの学習クラス（CNNベース）"""
    def __init__(self, num_agents, obs_dim, state_dim, action_dim,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 device=torch.device("cpu")):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # CNNエンコーダの共有
        self.cnn_encoder = CNNEncoder().to(device)

        # Actor/Criticネットワーク（CNNベース）
        self.actor = Actor(self.cnn_encoder, act_dim=action_dim).to(device)
        self.critic = Critic(self.cnn_encoder, num_agents=num_agents).to(device)

        # 最適化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # メモリバッファ
        self.buffer = MultiAgentBuffer(num_agents, obs_dim, state_dim, action_dim)

    def get_action(self, obs, greedy=False):
        """
        観測から行動とlog_probを取得

        Args:
            obs: (num_agents, obs_dim) のテンソル（CPU or GPU）
            greedy: Trueなら最大確率の行動を選択（評価用）

        Returns:
            actions: (num_agents,) の行動ID（CPU上のnumpy）
            log_probs: (num_agents,) のlog_prob（CPU上のnumpy）
        """
        # obsをdeviceに送る（呼び出し側で既にto(device)していても明示的に送る）
        obs = obs.to(self.device)

        with torch.no_grad():
            dist = self.actor(obs)  # (num_agents, action_dim) のlogitsを持つ分布

            if greedy:
                actions = torch.argmax(dist.logits, dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)
            # CPUに戻してnumpyで返す（バッファ保存用）
            return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_value(self, state):
        """
        グローバル状態から価値を取得

        Args:
            state: (state_dim,) のテンソル（CPU or GPU）

        Returns:
            value: スカラー（CPU上のfloat）
        """
        # stateをdeviceに送る
        state = state.to(self.device)

        with torch.no_grad():
            value = self.critic(state.unsqueeze(0))
            # CPUに戻してスカラーで返す（バッファ保存用）
            return value.item()

    def update(self, batch, epochs=3):
        """
        PPO更新（複数エポック）

        Args:
            batch: バッファからサンプリングしたミニバッチ（CPU上のnumpy/リスト）
            epochs: 更新エポック数
        """
        if batch is None:
            return

        # テンソル化してdeviceに送る
        obs = torch.FloatTensor(batch['obs']).to(self.device)           # (batch, num_agents, obs_dim)
        actions = torch.LongTensor(batch['actions']).to(self.device)     # (batch, num_agents)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)  # (batch, num_agents)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)  # (batch, num_agents)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)    # (batch, num_agents)
        global_states = torch.FloatTensor(batch['global_states']).to(self.device)  # (batch, state_dim)
        old_values = torch.FloatTensor(batch['values']).to(self.device)  # (batch,)

        # 正規化（オプション）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            # Actor更新（各エージェント分をまとめて処理）
            # obs: (batch, num_agents, obs_dim) -> (batch*num_agents, obs_dim)
            dist = self.actor(obs.view(-1, self.obs_dim))
            new_log_probs = dist.log_prob(actions.view(-1))  # (batch*num_agents,)
            entropy = dist.entropy().mean()

            # 確率比
            ratio = torch.exp(new_log_probs - old_log_probs.view(-1))

            # PPOのクリップ損失
            surr1 = ratio * advantages.view(-1)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.view(-1)
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            # Critic更新
            values = self.critic(global_states).squeeze()  # (batch,)
            value_loss = nn.MSELoss()(values, returns.mean(dim=1))  # グローバル報酬で学習

            self.optimizer_critic.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            # ログ出力（オプション）
            if epoch == 0:
                print(f"Epoch {epoch}: Actor Loss: {actor_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    def load_checkpoint(self, checkpoint_path):
        """
        MAPPOのチェックポイントを読み込み

        Args:
            checkpoint_path: チェックポイントファイルのパス
        Returns:
            episode: 保存時のエピソード番号
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # モデル・エンコーダの状態をロード
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])

        # 最適化器の状態をロード
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

        episode = checkpoint['episode']
        print(f"チェックポイントを読み込み: {checkpoint_path} (episode: {episode})")
        return episode