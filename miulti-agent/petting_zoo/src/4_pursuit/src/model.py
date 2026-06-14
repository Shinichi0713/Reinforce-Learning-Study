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


# MAPPO Centralized Critic: 全エージェントの観測（画像）と対象IDを集約して評価
class MAPPO_Critic(nn.Module):
    def __init__(self, cnn_encoder, num_agents=8, id_dim=8, hidden_size=256):
        super().__init__()
        # Actorと共有、または独立したCNNエンコーダ（MAPPOでは独立させることが多いです）
        self.cnn = cnn_encoder
        
        # エージェントIDの埋め込み層（「誰のための価値か」を識別するため）
        self.id_embedding = nn.Embedding(num_embeddings=num_agents, embedding_dim=id_dim)
        
        # 全エージェントのCNN特徴量の合計次元 + 対象エージェントのID次元
        total_feature_dim = (self.cnn.output_dim * num_agents) + id_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, all_obs, agent_id):
        """
        Args:
            all_obs: (batch_size, num_agents, 147) 全エージェントの共通/個別観測の集まり
            agent_id: (batch_size,) 状態価値を計算する対象のエージェントID
        """
        batch_size = all_obs.shape[0]
        num_agents = all_obs.shape[1]
        
        # 1. 全エージェントの全観測を効率よく一括でCNN処理するため、バッチ次元に平坦化
        # (batch_size * num_agents, 147)
        flat_obs = all_obs.view(-1, 147)
        flat_features = self.cnn(flat_obs)  # (batch_size * num_agents, 1568)
        
        # 2. 元のバッチとエージェントの形に戻し、全エージェントの特徴量を横に結合
        # (batch_size, num_agents * 1568)
        combined_features = flat_features.view(batch_size, -1)
        
        # 3. 評価対象エージェントのID特徴量を抽出
        if agent_id.dim() == 2:
            agent_id = agent_id.squeeze(-1)
        id_feats = self.id_embedding(agent_id)  # (batch_size, id_dim)
        
        # 4. 全員の特徴量と、対象のID特徴量を結合
        x = torch.cat([combined_features, id_feats], dim=-1)
        
        # 5. 状態価値 V(s) を出力
        value = self.mlp(x)
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

    def get_value(self, state):
        """
        グローバル状態（全員の観測の集まり）から価値を取得
        MAPPO本来の仕様に合わせ、各エージェント個別の価値を返せるように設計
        Args:
            state: (num_agents * obs_dim,) または (num_agents, obs_dim) のテンソル
        """
        # stateを (1, num_agents, obs_dim) に整形
        if state.dim() == 1:
            state = state.view(self.num_agents, self.obs_dim)
        state = state.unsqueeze(0).to(self.device) # バッチ次元を追加

        # ここでは環境全体の良さを代表して「エージェント0」の視点、
        # もしくはループ用に1つだけスカラーを返す元のインターフェースを維持
        dummy_agent_id = torch.tensor([0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            value = self.critic(state, dummy_agent_id)
            return value.item()

    def update(self, batch, epochs=3):
        if batch is None:
            return 0.0, 0.0, 0.0

        batch_size = batch['obs'].shape[0]

        obs = torch.FloatTensor(batch['obs']).to(self.device)                 # (batch, num_agents, obs_dim)
        actions = torch.LongTensor(batch['actions']).to(self.device)           # (batch, num_agents)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)   # (batch, num_agents)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)     # (batch, num_agents)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)           # (batch, num_agents)
        
        # グローバル状態（全員の観測の並び）
        # もし元のコードのバッファが(batch, num_agents * obs_dim)で保存していても、ここで復元
        global_states = torch.FloatTensor(batch['global_states']).to(self.device)
        if global_states.dim() == 2:
            global_states = global_states.view(batch_size, self.num_agents, self.obs_dim)

        # Advantageの正規化（バッチ次元のみ）
        advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (advantages.std(dim=0, keepdim=True) + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []

        # 更新で使い回すバッチサイズ分の基本IDシーケンス
        # shape: (batch_size, num_agents)
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
            # Centralized Criticでは全エージェント分ループ、あるいは一括処理を行います
            critic_epoch_losses = []
            
            for i in range(self.num_agents):
                # 対象エージェントIDのバッチテンソル
                target_agent_id = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
                
                # 画像情報を考慮した各エージェントの予測価値 V(s)
                values = self.critic(global_states, target_agent_id).squeeze(-1) # (batch,)
                
                # 各エージェント個別のリターンをターゲットにする
                target_returns = returns[:, i] # (batch,)
                
                critic_loss = nn.SmoothL1Loss()(values, target_returns)
                critic_epoch_losses.append(critic_loss)

            # エージェント全員分のCriticロスを合計して更新
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