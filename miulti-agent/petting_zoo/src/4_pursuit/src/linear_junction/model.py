import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

import torch
import torch.nn as nn
from torch.distributions import Categorical

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
        # CNN出力の次元（7*7*32 = 1568）
        self.output_dim = 7 * 7 * c2

    def forward(self, obs):
        """
        Args:
            obs: (batch_size, 147) のフラット観測テンソル
        Returns:
            features: (batch_size, cnn_output_dim) の特徴ベクトル
        """
        # (batch_size, 147) -> (batch_size, 7, 7, 3)
        obs = obs.view(-1, 7, 7, 3)
        # (batch_size, 7, 7, 3) -> (batch_size, 3, 7, 7)
        obs = obs.permute(0, 3, 1, 2)
        features = self.cnn(obs)
        return features


# MAPPO Actor: ID情報を付与し、パラメータを共有して使用
class MAPPO_Actor(nn.Module):
    def __init__(self, cnn_encoder, num_agents=8, id_dim=8, act_dim=5, hidden_size=256):
        super().__init__()
        self.cnn = cnn_encoder

        # エージェントIDの埋め込み層
        self.id_embedding = nn.Embedding(num_embeddings=num_agents, embedding_dim=id_dim)

        # 入力サイズ = CNN特徴量(1568) + ID埋め込み(8)
        input_dim = self.cnn.output_dim + id_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs, agent_id):
        """
        Args:
            obs: (batch_size, 147) 単一エージェントの観測
            agent_id: (batch_size,) 単一エージェントのID
        """
        # 空間特徴量の抽出
        features = self.cnn(obs)

        # ID特徴量の抽出
        if agent_id.dim() == 2:
            agent_id = agent_id.squeeze(-1)
        id_feats = self.id_embedding(agent_id)

        # 特徴量の結合
        x = torch.cat([features, id_feats], dim=-1)

        # 行動分布の出力
        logits = self.mlp(x)
        dist = Categorical(logits=logits)
        return dist

class MAPPO_Critic(nn.Module):
    def __init__(self, cnn_encoder, num_agents, agent_emb_dim=16):
        super(MAPPO_Critic, self).__init__()
        self.num_agents = num_agents
        self.cnn_encoder = cnn_encoder 
        
        # 🌟 【重要修正】cnn_encoder が配置されているデバイス（CPU or GPU）を自動取得
        device = next(cnn_encoder.parameters()).device
        
        # ダミーデータを使って、CNNEncoder の本当の出力サイズを自動計測する
        with torch.no_grad():
            # 🛠️ ダミーテンソルを cnn_encoder と同じデバイス（GPU）に送る
            dummy_obs = torch.zeros(1, 147, device=device)
            dummy_features = self.cnn_encoder(dummy_obs)
            # 実際の出力次元を取得
            self.cnn_feature_dim = dummy_features.shape[-1]
            
        print(f"[MAPPO_Critic] 自動検出されたCNN出力次元: {self.cnn_feature_dim}")
        
        # ターゲットエージェントのID埋め込み
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)
        
        # 全員の特徴量（自動計測値 * num_agents） + 対象エージェントID（agent_emb_dim）
        input_dim = (self.cnn_feature_dim * num_agents) + agent_emb_dim
        
        # 特徴ベクトルの集まりを処理するMLP
        self.v_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_states, target_agent_id):
        # デバイスの統一（引数がCPUのままの場合に備える）
        device = next(self.parameters()).device
        global_states = global_states.to(device)
        target_agent_id = target_agent_id.to(device)

        # 形状の安全な復元
        if global_states.dim() == 1:
            # 1176次次元を (8, 147) に復元
            global_states = global_states.view(self.num_agents, -1)
        if global_states.dim() == 2:
            global_states = global_states.unsqueeze(0) # (1, 8, 147)

        batch_size = global_states.size(0)
        
        # 1. 各エージェントの観測を個別にCNNに通す
        flat_obs = global_states.view(batch_size * self.num_agents, -1) 
        agent_features = self.cnn_encoder(flat_obs) # (batch*num_agents, 1568)
        
        # 2. 全員分の特徴量を1つのバッチベクトルに再結合
        # 自動計測したサイズを使うため、12544個の要素が綺麗に (batch_size, 12544) に収まります
        combined_features = agent_features.view(batch_size, self.num_agents * self.cnn_feature_dim)
        
        # 3. 評価対象エージェントのID情報を結合
        if target_agent_id.dim() == 0:
            target_agent_id = target_agent_id.unsqueeze(0)
            
        agent_emb = self.agent_embedding(target_agent_id) 
        x = torch.cat([combined_features, agent_emb], dim=-1) 
        
        # 4. 価値 V(s) の算出
        value = self.v_head(x)
        return value


# =====================================================================
# 2. MAPPO トレーナークラス
# =====================================================================
class MAPPO:
    def __init__(self, num_agents, obs_dim, state_dim, action_dim,
                 lr_actor=3e-5, lr_critic=3e-5, gamma=0.99, gae_lambda=0.95,
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

        # CNNエンコーダ（Actor/Criticで独立して管理）
        self.cnn_encoder = CNNEncoder().to(device)
        self.critic_encoder = CNNEncoder().to(device) # 画像処理用Criticエンコーダ

        # ID情報に対応したActorとCritic
        self.actor = MAPPO_Actor(self.cnn_encoder, num_agents=num_agents, act_dim=action_dim).to(device)
        self.critic = MAPPO_Critic(self.critic_encoder, num_agents=num_agents).to(device)

        # 最適化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = MultiAgentBuffer(num_agents, obs_dim, state_dim, action_dim)

    def get_action(self, obs, greedy=False):
        """
        観測から行動とlog_probを取得
        Args:
            obs: (num_agents, obs_dim) のテンソル
        """
        obs = obs.to(self.device)

        # 固有IDテンソルの作成 [0, 1, 2, ..., num_agents-1]
        agent_ids = torch.arange(self.num_agents, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # ID情報を付与してポリシーに送る
            dist = self.actor(obs, agent_ids)

            if greedy:
                actions = torch.argmax(dist.logits, dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)

            return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_value_for_agent(self, state, agent_id):
        """
        🌟 追加: 特定のエージェント視点での中央集中型Criticの価値を算出する
        学習ループ内での個々の努力評価（ハイブリッド報酬の計算・蓄積）に使用します。

        Args:
            state: (1, num_agents, obs_dim) または (num_agents * obs_dim,) のテンソル
            agent_id: torch.tensor([i], dtype=torch.long)
        """
        # 形状の安全な復元
        if state.dim() == 1:
            state = state.view(self.num_agents, self.obs_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0) # (1, num_agents, obs_dim)

        state = state.to(self.device)
        agent_id = agent_id.to(self.device)

        with torch.no_grad():
            value = self.critic(state, agent_id)
            return value.item()

    def get_value(self, state):
        """
        互換性のために維持: 従来のグローバル状態から代表（エージェント0）の価値を返す
        Args:
            state: (num_agents * obs_dim,) または (num_agents, obs_dim) のテンソル
        """
        dummy_agent_id = torch.tensor([0], dtype=torch.long, device=self.device)
        return self.get_value_for_agent(state, dummy_agent_id)

    def update(self, batch, epochs=3):
        if batch is None:
            return 0.0, 0.0, 0.0

        batch_size = batch['obs'].shape[0]

        obs = torch.FloatTensor(batch['obs']).to(self.device)                  # (batch, num_agents, obs_dim)
        actions = torch.LongTensor(batch['actions']).to(self.device)           # (batch, num_agents)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)   # (batch, num_agents)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)     # (batch, num_agents)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)           # (batch, num_agents) ※実質は個別リターン

        # グローバル状態（全員の観測の並び）
        global_states = torch.FloatTensor(batch['global_states']).to(self.device)

        # 🛠️ 変更点: Criticに (batch_size, num_agents, obs_dim) の形状で綺麗に渡すため、
        # もしバッファから2次元で出てきても確実に3次元(B, N, D)に整形する
        if global_states.dim() == 2:
            global_states = global_states.view(batch_size, self.num_agents, self.obs_dim)

        # Advantageの正規化（バッチ次元のみ）
        advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (advantages.std(dim=0, keepdim=True) + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []

        # 更新で使い回すバッチサイズ分の基本IDシーケンス
        batch_agent_ids = torch.arange(self.num_agents, dtype=torch.long, device=self.device).repeat(batch_size, 1)

        for epoch in range(epochs):
            # --- 1. Actorの更新 ---
            flat_obs = obs.view(-1, self.obs_dim)
            flat_ids = batch_agent_ids.view(-1)

            # IDを付与してフォワードパス
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
                
                # 🛠️ 変更点: 綺麗にセパレートされたglobal_statesが、新設したCriticのMLPベースのフォワードに入ります
                values = self.critic(global_states, target_agent_id).squeeze(-1) # (batch,)
                
                target_returns = returns[:, i] # (batch,)
                
                critic_loss = nn.SmoothL1Loss()(values, target_returns)
                critic_epoch_losses.append(critic_loss)

            # エージェント全員分のCriticロスを合計し、係数を掛け合わせて更新
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
        self.cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder_state_dict'])

        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

        episode = checkpoint['episode']
        print(f"チェックポイントを読み込み: {checkpoint_path} (episode: {episode})")
        return episode