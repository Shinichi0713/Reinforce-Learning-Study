import torch
import torch.nn as nn
from torch.distributions import Categorical

class WizardOfWorMAPPONet(nn.Module):
    def __init__(self, in_channels, action_space_n, num_agents=2, is_critic=False):
        super(WizardOfWorMAPPONet, self).__init__()
        
        # 1. CNN Encoder: 入力チャンネル数(in_channels)を柔軟に受け取る
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # ダミー入力で flattening サイズを計算
        with torch.no_grad():
            # 形状は (1, C, 84, 84) 等を想定
            dummy_input = torch.zeros(1, in_channels, 84, 84) 
            n_flatten = self.cnn_encoder(dummy_input).shape[1]
        
        self.shared_fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU()
        )
        
        # Head定義
        self.head = nn.Linear(512 + num_agents, 1 if is_critic else action_space_n)

    def forward(self, x, agent_id_onehot):
        # 防御策: 画像(4次元)以外がIDに入ってきた時のチェックは継続
        if agent_id_onehot.dim() == 4:
            raise RuntimeError(f"引数順エラー: IDに4次元テンソルが渡されました。")

        cnn_features = self.cnn_encoder(x)
        latent = self.shared_fc(cnn_features)
        
        if agent_id_onehot.dim() == 1:
            agent_id_onehot = agent_id_onehot.unsqueeze(0)
            
        combined = torch.cat([latent, agent_id_onehot], dim=-1)
        return self.head(combined)

class MAPPO_ActorCritic(nn.Module):
    def __init__(self, obs_shape, state_shape, action_space_n, num_agents, device="cpu"):
        super().__init__()
        self.obs_shape = obs_shape  # (C, H, W)
        self.state_shape = state_shape  # (C_total, H, W)
        self.action_space_n = action_space_n
        self.num_agents = num_agents
        self.device = device

        # CNN エンコーダ（観測画像の特徴抽出）
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # CNN 出力サイズを計算
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *obs_shape)
            cnn_out_dim = self.cnn_encoder(dummy_obs).shape[-1]

        # Actor ヘッド（行動確率分布）
        self.actor_fc = nn.Sequential(
            nn.Linear(cnn_out_dim + num_agents, 512),  # CNN特徴 + エージェントID
            nn.ReLU(),
            nn.Linear(512, action_space_n)
        )

        # Critic ヘッド（状態価値）
        self.critic_fc = nn.Sequential(
            nn.Linear(cnn_out_dim + num_agents, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, state, agent_id_onehot):
        """
        x: (batch, C, H, W) 各エージェントの観測
        state: (batch, C_total, H, W) 集中状態（全観測結合）
        agent_id_onehot: (batch, num_agents) エージェントIDのone-hot
        """
        # CNN で観測特徴を抽出
        cnn_features = self.cnn_encoder(x)  # (batch, cnn_out_dim)

        # CNN特徴 + エージェントID を結合
        features = torch.cat([cnn_features, agent_id_onehot], dim=-1)  # (batch, cnn_out_dim + num_agents)

        # Actor: 行動確率分布
        logits = self.actor_fc(features)  # (batch, action_space_n)
        dist = torch.distributions.Categorical(logits=logits)

        # Critic: 状態価値
        value = self.critic_fc(features)  # (batch, 1)

        return value, dist

    def get_actions(self, obs, state, agent_id_onehot=None):
        """
        推論用: 行動と価値をサンプリング
        obs: (num_agents, C, H, W)
        state: (num_agents, C_total, H, W)
        agent_id_onehot: (num_agents, num_agents) の one-hot
        """
        if agent_id_onehot is None:
            agent_id_onehot = torch.eye(self.num_agents, device=self.device)

        value, dist = self.forward(obs, state, agent_id_onehot)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return value, action, action_log_prob

    def evaluate_actions(self, obs, state, actions, agent_id_onehot=None):
        """
        学習用: 指定された行動の対数確率とエントロピーを計算
        obs: (batch, C, H, W)
        state: (batch, C_total, H, W)
        actions: (batch, 1)
        agent_id_onehot: (batch, num_agents)
        """
        if agent_id_onehot is None:
            batch_size = obs.size(0)
            agent_id_onehot = torch.eye(self.num_agents, device=self.device).repeat(batch_size // self.num_agents, 1)

        value, dist = self.forward(obs, state, agent_id_onehot)
        action_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    # def get_value(self, state):
    #     """
    #     状態価値のみを計算（GAE計算用）
    #     state: (num_agents, C_total, H, W)
    #     """
    #     agent_id_onehot = torch.eye(self.num_agents, device=self.device)
    #     value, _ = self.forward(state, state, agent_id_onehot)
    #     return value
    def get_value(self, state, agent_id_onehot):
        """
        修正後: 入力された state と agent_id を使って価値を計算する
        """
        # forward は (obs, state, id) を取るが、Criticのみ計算する場合は
        # obsの代わりにダミー(stateの一部)か、あるいはForwardを適宜調整する必要がある。
        # 現在の forward 実装に合わせるなら以下：
        
        # obsとして state[:, :3, :, :] (エージェント1の画像) を代入する暫定対応
        value, _ = self.forward(state[:, :3, :, :], state, agent_id_onehot)
        return value


import torch
import numpy as np

class SharedRolloutBuffer:
    def __init__(self, num_agents, buffer_size, obs_shape, state_shape, device="cpu"):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.device = device

        # --- メモリ節約のため画像系は uint8 で保持 ---
        self.obs = torch.zeros((buffer_size + 1, num_agents, *obs_shape), dtype=torch.uint8)
        self.state = torch.zeros((buffer_size + 1, num_agents, *state_shape), dtype=torch.uint8)
        
        # --- 報酬・行動・数値系は float32/long ---
        self.actions = torch.zeros((buffer_size, num_agents, 1), dtype=torch.long)
        self.log_probs = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        self.values = torch.zeros((buffer_size + 1, num_agents, 1), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        self.masks = torch.ones((buffer_size + 1, num_agents, 1), dtype=torch.float32)
        
        # 計算結果用
        self.returns = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        self.advantages = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        
        self.step = 0

    def insert_first(self, obs, state):
        """初期状態の挿入（環境リセット時）"""
        self.obs[0] = self._to_uint8_tensor(obs)
        self.state[0] = self._to_uint8_tensor(state)

    def insert(self, next_obs, next_state, actions, log_probs, values, rewards, masks):
        """1ステップのデータを挿入"""
        self.obs[self.step + 1] = self._to_uint8_tensor(next_obs)
        self.state[self.step + 1] = self._to_uint8_tensor(next_state)
        
        # actions を (num_agents, 1) に整形
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)  # (num_agents,) -> (num_agents, 1)
        self.actions[self.step] = actions.clone()
        
        # log_probs も同様に整形
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(-1)
        self.log_probs[self.step] = log_probs.clone()
        
        # values も同様に整形
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        self.values[self.step] = values.clone()
        
        self.rewards[self.step] = rewards.clone()
        self.masks[self.step + 1] = masks.clone()

        self.step = (self.step + 1) % self.buffer_size

    def compute_returns(self, next_value, gamma=0.99, gae_lambda=0.95):
        """GAEを用いたアドバンテージとリターンの計算"""
        self.values[-1] = next_value.clone()
        gae = 0
        for step in reversed(range(self.buffer_size)):
            # ベルマン誤差 (delta) の計算
            delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.advantages[step] = gae
            self.returns[step] = self.advantages[step] + self.values[step]

    def after_update(self):
        """最新の観測を次のエピソードの開始点（index 0）にコピー"""
        self.obs[0].copy_(self.obs[-1])
        self.state[0].copy_(self.state[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    def get_generator(self, advantages, num_mini_batches):
        """学習用のミニバッチを生成（ここで float 化と正規化を行う）"""
        batch_size = self.buffer_size * self.num_agents
        mini_batch_size = batch_size // num_mini_batches
        
        # データのフラット化
        flat_obs = self.obs[:-1].reshape(-1, *self.obs_shape)
        flat_state = self.state[:-1].reshape(-1, *self.state_shape)
        flat_actions = self.actions.reshape(-1, 1)
        flat_log_probs = self.log_probs.reshape(-1, 1)
        flat_returns = self.returns.reshape(-1, 1)
        flat_adv = advantages.reshape(-1, 1)
        flat_masks = self.masks[:-1].reshape(-1, 1)
        
        # Agent ID (One-hot)
        flat_ids = torch.eye(self.num_agents).repeat(self.buffer_size, 1)

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            
            # ここで float に変換し、0-1 に正規化して yield
            yield (
                flat_obs[idx].to(self.device).float() / 255.0,
                flat_state[idx].to(self.device).float() / 255.0,
                flat_actions[idx].to(self.device),
                flat_log_probs[idx].to(self.device),
                flat_returns[idx].to(self.device),
                flat_adv[idx].to(self.device),
                flat_masks[idx].to(self.device),
                flat_ids[idx].to(self.device)
            )

    def _to_uint8_tensor(self, data):
        """あらゆる入力形式を uint8 の Tensor に安全に変換する"""
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array(data))
        
        # float 型の場合は 0-255 にスケールしてから uint8 に変換
        if data.is_floating_point():
            # 0-1 に正規化されている前提で 255 倍
            data = (data * 255.0).clamp(0, 255)
        
        return data.to(torch.uint8)
    
    def _to_uint8_tensor(self, data):
        """あらゆる入力形式を uint8 の Tensor に安全に変換する"""
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array(data))
        
        # float 型の場合は 0-255 にスケールしてから uint8 に変換
        if data.is_floating_point():
            # 0-1 に正規化されている前提で 255 倍
            data = (data * 255.0).clamp(0, 255)
        
        return data.to(torch.uint8)

    def get_obs_step(self, step=None):
        """指定ステップの観測を float 型・正規化済みで取得（推論用）"""
        s = self.step if step is None else step
        # (num_agents, C, H, W) の float テンソルを返す
        return self.obs[s].float() / 255.0, self.state[s].float() / 255.0