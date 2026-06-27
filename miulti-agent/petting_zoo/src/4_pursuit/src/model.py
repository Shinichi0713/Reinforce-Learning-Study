import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

# =====================================================================
# 1. Transformer アーキテクチャ
# =====================================================================

class MAPPO_TransformerActor(nn.Module):
    def __init__(self, obs_range=7, in_channels=4, d_model=64, nhead=4, num_layers=2, act_dim=5, num_agents=8, id_dim=8, hidden_size=256):
        super().__init__()
        self.obs_range = obs_range
        self.num_tokens = obs_range * obs_range  # 7x7 = 49

        # 1. 各マスの4次元情報を d_model(64次元) に引き上げる線形埋め込み
        self.embedding = nn.Linear(in_channels, d_model)

        # 2. 2次元空間用の学習可能な位置エンコーディング
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # 3. エージェント固有IDの埋め込み（譲り合いの個性を学習）
        self.id_embedding = nn.Embedding(num_embeddings=num_agents, embedding_dim=id_dim)

        # 4. Transformer Encoder（GELUを内部で採用）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 出力用MLP：中心セルの特徴量（64） + エージェントID（8）
        input_dim = d_model + id_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs, agent_id):
        """
        Args:
            obs: (batch_size, 196) のフラットな4チャンネル観測
            agent_id: (batch_size,) のエージェント固有ID
        """
        batch_size = obs.shape[0]

        # (batch_size, 196) -> (batch_size, 49, 4) へトークン変形
        x = obs.view(batch_size, self.num_tokens, 4)

        # トークン埋め込み + 位置エンコーディング
        x = self.embedding(x) + self.pos_embedding  # (batch_size, 49, 64)

        # Transformer処理
        features = self.transformer(x)  # (batch_size, 49, 64)

        # 視界の中心(3,3) ＝ インデックス 24 (3*7 + 3) の自身のトークン特徴量を抽出
        my_feature = features[:, 24, :]  # (batch_size, 64)

        # ID特徴量の統合
        if agent_id.dim() == 2:
            agent_id = agent_id.squeeze(-1)
        id_feats = self.id_embedding(agent_id)  # (batch_size, 8)

        # 結合して行動のカテゴリカル分布を返却
        combined = torch.cat([my_feature, id_feats], dim=-1)
        logits = self.mlp(combined)

        return Categorical(logits=logits)


class MAPPO_TransformerCritic(nn.Module):
    def __init__(self, num_agents=8, obs_range=7, in_channels=4, d_model=64, nhead=4, num_layers=2, agent_emb_dim=16):
        super().__init__()
        self.num_agents = num_agents
        self.num_tokens_per_agent = obs_range * obs_range  # 49
        self.total_tokens = self.num_tokens_per_agent * num_agents  # 49 * 8 = 392

        # 各エージェントマスの特徴抽出埋め込み
        self.embedding = nn.Linear(in_channels, d_model)

        # 全トークン（392個）の位置・所属エージェント認識用エンコーディング
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_tokens, d_model))

        # 評価対象ターゲットエージェントのID埋め込み
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 統合価値出力ヘッド
        self.v_head = nn.Sequential(
            nn.Linear(d_model + agent_emb_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_states, target_agent_id):
        """
        Args:
            global_states: (batch_size, num_agents, 196) またはフラットなテンソル
            target_agent_id: (batch_size,) の対象エージェントID
        """
        device = next(self.parameters()).device
        global_states = global_states.to(device)
        target_agent_id = target_agent_id.to(device)

        # 形状の安全な復元 (batch_size, num_agents, 196)
        if global_states.dim() == 1:
            global_states = global_states.view(1, self.num_agents, -1)
        elif global_states.dim() == 2:
            global_states = global_states.view(-1, self.num_agents, self.num_tokens_per_agent * 4)

        batch_size = global_states.size(0)

        # 全員分を1つの巨大なトークンシーケンスに変形 (batch_size, 392, 4)
        x = global_states.view(batch_size, self.total_tokens, 4)

        # 埋め込みと自己アテンション処理
        x = self.embedding(x) + self.pos_embedding
        features = self.transformer(x)  # (batch_size, 392, 64)

        # 平均プーリングによるグローバル盤面表現の圧縮
        global_feature = torch.mean(features, dim=1)  # (batch_size, 64)

        # 評価ターゲットIDの結合
        if target_agent_id.dim() == 2:
            target_agent_id = target_agent_id.squeeze(-1)
        elif target_agent_id.dim() == 0:
            target_agent_id = target_agent_id.unsqueeze(0).repeat(batch_size)

        agent_emb = self.agent_embedding(target_agent_id)

        combined = torch.cat([global_feature, agent_emb], dim=-1)
        return self.v_head(combined)


# =====================================================================
# 2. MAPPO トレーナークラス
# =====================================================================
class MAPPO:
    def __init__(self, num_agents, obs_dim=196, state_dim=1568, action_dim=5,
                 lr_actor=1e-4, lr_critic=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 device=torch.device("cpu")):
        self.num_agents = num_agents
        self.obs_dim = obs_dim  # 🌟 196 (7x7x4)
        self.state_dim = state_dim  # 🌟 1568 (196x8)
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # 🌟 旧CNNエンコーダを完全に廃止し、Transformerベースのモデルを初期化
        self.actor = MAPPO_TransformerActor(num_agents=num_agents, act_dim=action_dim).to(device)
        self.critic = MAPPO_TransformerCritic(num_agents=num_agents).to(device)

        # 最適化器 (Transformerの学習安定化のため、デフォルト学習率を 1e-4 に引き上げて調整)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ※バッファの初期化（外部で定義されている MultiAgentBuffer を利用前提）
        # self.buffer = MultiAgentBuffer(num_agents, obs_dim, state_dim, action_dim)

    def get_action(self, obs, greedy=False):
        """
        Args:
            obs: (num_agents, 196) のテンソル
        """
        obs = obs.to(self.device)
        agent_ids = torch.arange(self.num_agents, dtype=torch.long, device=self.device)

        with torch.no_grad():
            dist = self.actor(obs, agent_ids)

            if greedy:
                actions = torch.argmax(dist.logits, dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)

            return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_value_for_agent(self, state, agent_id):
        """
        特定のエージェント視点での中央集中型Transformer Criticの価値を算出
        """
        if state.dim() == 1:
            state = state.view(self.num_agents, self.obs_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0)

        state = state.to(self.device)
        agent_id = agent_id.to(self.device)

        with torch.no_grad():
            value = self.critic(state, agent_id)
            return value.item()

    def get_value(self, state):
        dummy_agent_id = torch.tensor([0], dtype=torch.long, device=self.device)
        return self.get_value_for_agent(state, dummy_agent_id)

    def update(self, batch, epochs=3):
        if batch is None:
            return 0.0, 0.0, 0.0

        batch_size = batch['obs'].shape[0]

        obs = torch.FloatTensor(batch['obs']).to(self.device)                  # (batch, num_agents, 196)
        actions = torch.LongTensor(batch['actions']).to(self.device)           # (batch, num_agents)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)   # (batch, num_agents)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)     # (batch, num_agents)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)           # (batch, num_agents)

        global_states = torch.FloatTensor(batch['global_states']).to(self.device)

        if global_states.dim() == 2:
            global_states = global_states.view(batch_size, self.num_agents, self.obs_dim)

        # Advantageの正規化
        advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (advantages.std(dim=0, keepdim=True) + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []

        batch_agent_ids = torch.arange(self.num_agents, dtype=torch.long, device=self.device).repeat(batch_size, 1)

        for epoch in range(epochs):
            # --- 1. Actorの更新 ---
            flat_obs = obs.view(-1, self.obs_dim)
            flat_ids = batch_agent_ids.view(-1)

            dist = self.actor(flat_obs, flat_ids)

            new_log_probs = dist.log_prob(actions.view(-1)).view(batch_size, self.num_agents)
            entropy = dist.entropy().view(batch_size, self.num_agents).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            # --- 2. Criticの更新 ---
            critic_epoch_losses = []

            for i in range(self.num_agents):
                target_agent_id = torch.full((batch_size,), i, dtype=torch.long, device=self.device)

                # Transformer Criticへ整形した状態データをフォワード
                values = self.critic(global_states, target_agent_id).squeeze(-1)  # (batch,)
                target_returns = returns[:, i]

                critic_loss = nn.SmoothL1Loss()(values, target_returns)
                critic_epoch_losses.append(critic_loss)

            total_critic_loss = torch.stack(critic_epoch_losses).mean() * self.value_coef

            self.optimizer_critic.zero_grad()
            total_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(total_critic_loss.item())
            entropies.append(entropy.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

        episode = checkpoint['episode']
        print(f"チェックポイントを読み込み: {checkpoint_path} (episode: {episode})")
        return episode