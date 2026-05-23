import torch
import torch.nn as nn
import torch.nn.functional as F

class WizardOfWorMAPPONet(nn.Module):
    def __init__(self, action_space_n, num_agents=2):
        super(WizardOfWorMAPPONet, self).__init__()
        
        # 1. 共通の CNN Encoder (共通の「目」)
        # 入力画像サイズ: (3, 210, 160) を想定
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN出力次元の計算 (210x160入力の場合、64*22*16 = 22528次元程度)
        # shared_fc を通して 512次元に圧縮
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 22 * 16, 512),
            nn.ReLU()
        )
        
        # 2. Actor Head (分散実行用の「脳」)
        # 入力: 512 (画像特徴) + num_agents (Agent ID の One-hot)
        self.actor_head = nn.Sequential(
            nn.Linear(512 + num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_n) # 行動ごとのロジットを出力
        )
        
        # 3. Critic Head (集中学習用の「司令塔」)
        # MAPPOではCriticもIDを受け取り「そのエージェントから見た価値」を計算
        self.critic_head = nn.Sequential(
            nn.Linear(512 + num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 状態価値 V(s) を出力
        )

    def forward(self, obs, agent_id_onehot):
        """
        引数:
            obs: 画像テンソル [Batch, 3, 210, 160] (0.0~1.0に正規化済み)
            agent_id_onehot: IDのOne-hotテンソル [Batch, num_agents]
        """
        # 共通バックボーンで特徴抽出
        cnn_features = self.cnn_encoder(obs)
        latent = self.shared_fc(cnn_features)
        
        # 特徴ベクトルと Agent ID を結合 (Concatenate)
        # [Batch, 512] + [Batch, 2] -> [Batch, 514]
        combined = torch.cat([latent, agent_id_onehot], dim=-1)
        
        # Actor: 行動分布 (Categorical分布の作成用にロジットを返す)
        action_logits = self.actor_head(combined)
        
        # Critic: 状態価値
        state_value = self.critic_head(combined)
        
        return action_logits, state_value

    def get_action(self, obs, agent_id_onehot, deterministic=False):
        """推論時に行動を選択するためのヘルパーメソッド"""
        action_logits, _ = self.forward(obs, agent_id_onehot)
        probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            # 確率分布からサンプリング
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
        return action.item()