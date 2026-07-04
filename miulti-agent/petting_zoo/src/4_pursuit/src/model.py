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
        self.spatial_dim = obs_range * obs_range * in_channels  # 196
        self.num_agents = num_agents

        # 1. 各マスの4次元情報を d_model(64次元) に引き上げる線形埋め込み
        self.embedding = nn.Linear(in_channels, d_model)

        # 2. 2次元空間用の学習可能な位置エンコーディング
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # 🌟 変更点1: 他エージェントの行動履歴（40次元）を処理する線形層
        # 8人分×5行動を埋め込み、Transformerの各トークンに付加的なコンテキストとして加算/結合できるようにする
        self.action_history_embed = nn.Sequential(
            nn.Linear(num_agents * 5, d_model),
            nn.GELU()
        )

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
            obs: (batch_size, 236) のフラットな観測 (196次元空間 + 40次元行動履歴)
            agent_id: (batch_size,) のエージェント固有ID
        """
        batch_size = obs.shape[0]

        # 🌟 変更点2: 空間特徴量(196) と 行動履歴履歴(40) を分離
        spatial_obs = obs[:, :self.spatial_dim]          # (batch_size, 196)
        action_history = obs[:, self.spatial_dim:]       # (batch_size, 40)

        # (batch_size, 196) -> (batch_size, 49, 4) へトークン変形
        x = spatial_obs.view(batch_size, self.num_tokens, -1)

        # トークン埋め込み + 位置エンコーディング
        x = self.embedding(x) + self.pos_embedding  # (batch_size, 49, 64)

        # 🌟 変更点3: 行動履歴の埋め込みをブロードキャストして、空間トークンにコンテキストとして加算
        # 全員が「直前誰がどう動いたか」を知った上で、各マスのAttentionを計算できるようにします
        act_emb = self.action_history_embed(action_history).unsqueeze(1)  # (batch_size, 1, 64)
        x = x + act_emb  # 空間特徴に協調コンテキストをブレンド

        # Transformer処理
        features = self.transformer(x)  # (batch_size, 49, 64)

        # 視界の中心(3,3) ＝ インデックス 24 の自身のトークン特徴量を抽出
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
        self.spatial_dim_per_agent = obs_range * obs_range * in_channels  # 196
        self.obs_dim_per_agent = self.spatial_dim_per_agent + (num_agents * 5)  # 236
        self.total_tokens = self.num_tokens_per_agent * num_agents  # 49 * 8 = 392

        # 各エージェントマスの特徴抽出埋め込み
        self.embedding = nn.Linear(in_channels, d_model)

        # 全トークン（392個）の位置・所属エージェント認識用エンコーディング
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_tokens, d_model))

        # 🌟 変更点4: Critic側でも全員の行動履歴（40次元）を集約するレイヤー
        self.global_action_embed = nn.Sequential(
            nn.Linear(num_agents * 5, d_model),
            nn.GELU()
        )

        # 評価対象ターゲットエージェントのID埋め込み
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 統合価値出力ヘッド（グローバル行動特徴も考慮するため、入力を拡張）
        self.v_head = nn.Sequential(
            nn.Linear(d_model * 2 + agent_emb_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_states, target_agent_id):
        """
        Args:
            global_states: (batch_size, num_agents, 236) のテンソル
            target_agent_id: (batch_size,) の対象エージェントID
        """
        device = next(self.parameters()).device
        global_states = global_states.to(device)
        target_agent_id = target_agent_id.to(device)

        batch_size = global_states.size(0)

        # 🌟 変更点5: 1エージェントあたり236次元に拡張された global_states を空間と行動履歴に分離
        # global_statesの形状: (batch_size, 8, 236)
        spatial_states = global_states[:, :, :self.spatial_dim_per_agent]  # (batch_size, 8, 196)
        
        # 行動履歴は共通バッファなため、インデックス0番のエージェントの観測末尾から代表して抽出
        action_history = global_states[:, 0, self.spatial_dim_per_agent:]  # (batch_size, 40)

        # 全員分を1つの巨大なトークンシーケンスに変形 (batch_size, 392, 4)
        x = spatial_states.reshape(batch_size, self.total_tokens, -1)

        # 埋め込みと自己アテンション処理
        x = self.embedding(x) + self.pos_embedding
        features = self.transformer(x)  # (batch_size, 392, 64)

        # 平均プーリングによるグローバル盤面表現の圧縮
        global_spatial_feature = torch.mean(features, dim=1)  # (batch_size, 64)

        # 🌟 変更点6: グローバル行動履歴の埋め込み
        global_action_feature = self.global_action_embed(action_history)  # (batch_size, 64)

        # 評価ターゲットIDの結合
        if target_agent_id.dim() == 2:
            target_agent_id = target_agent_id.squeeze(-1)
        elif target_agent_id.dim() == 0:
            target_agent_id = target_agent_id.unsqueeze(0).repeat(batch_size)

        agent_emb = self.agent_embedding(target_agent_id)

        # 空間特徴 + 行動特徴 + ターゲットID特徴をすべて結合
        combined = torch.cat([global_spatial_feature, global_action_feature, agent_emb], dim=-1)
        return self.v_head(combined)


# =====================================================================
# 2. MAPPO トレーナークラス
# =====================================================================
class MAPPO:
    def __init__(self, num_agents, obs_dim=236, state_dim=1888, action_dim=5,
                 lr_actor=1e-4, lr_critic=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 device=torch.device("cpu")):
        self.num_agents = num_agents
        self.obs_dim = obs_dim  # 🌟 236 に拡張 (196 + 40)
        self.state_dim = state_dim  # 🌟 1888 に拡張 (236x8)
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # Transformerベースのモデルを初期化
        self.actor = MAPPO_TransformerActor(num_agents=num_agents, act_dim=action_dim).to(device)
        self.critic = MAPPO_TransformerCritic(num_agents=num_agents).to(device)

        # 最適化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_action(self, obs, greedy=False):
        """
        Args:
            obs: (num_agents, 236) のテンソル
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
        # 🌟 変更点7: 形状の安全な復元次元を obs_dim=236 ベースに修正
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

        obs = torch.FloatTensor(batch['obs']).to(self.device)                  # (batch, num_agents, 236)
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