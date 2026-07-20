import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ユーザーが提示した TransformerQNetwork（Flatten方式）
# もしこちらを試す場合は、エージェント側の _preprocess_state で
# permute や reshape をせず、(Batch, 4, rows, cols) のテンソルをそのまま送る必要があります。
class TransformerQNetworkV2(nn.Module):
    def __init__(self, in_channels=4, grid_size=5, d_model=64, nhead=4, num_layers=2, action_dim=4, hidden_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels

        self.embedding = nn.Linear(in_channels, d_model)

        # 可変な迷路サイズにも柔軟に対応できるよう、学習可能な位置エンコーディングにする
        self.pos_embedding = nn.Parameter(torch.randn(1, grid_size * grid_size, d_model))
        # CLS（エージェントの意思決定用）トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力は CLS トークンの d_model 分だけで受けるため、軽量かつ汎化性能が高い
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )

        # 出力の初期化（Q値の初期のブレを抑える）
        # with torch.no_grad():
        #     self.mlp[-1].weight.fill_(0.0)
        #     self.mlp[-1].bias.fill_(0.0)

    def forward(self, x):
        # x: (Batch, num_tokens, in_channels)
        B = x.shape[0]

        x = self.embedding(x)
        x = x + self.pos_embedding  # 空間位置情報を付加

        # CLS トークンをバッチサイズ分に拡張して先頭に結合
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch, 1 + num_tokens, d_model)

        # Transformer で大域的な依存関係（ゴールへの経路など）を計算
        x = self.transformer(x)

        # 先頭の CLS トークンのみを抽出（全マスの状況を集約したベクトルになっている）
        cls_out = x[:, 0]  # (Batch, d_model)

        return self.mlp(cls_out)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class TransformerDQNAgent:
    def __init__(
        self,
        env,  # MazeEnv インスタンス
        in_channels=4,
        grid_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        action_dim=4,
        hidden_size=128,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        buffer_capacity=10000,
        batch_size=32,
        target_update_interval=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device

        # Qネットワーク（オンライン）
        self.q_online = TransformerQNetworkV2(
            in_channels=in_channels,
            grid_size=grid_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_dim=action_dim,
            hidden_size=hidden_size
        ).to(device)

        # ターゲットネットワーク
        self.q_target = TransformerQNetworkV2(
            in_channels=in_channels,
            grid_size=grid_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_dim=action_dim,
            hidden_size=hidden_size
        ).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.step_count = 0

        self.temperature = epsilon_start      # 初期温度（例: 1.0）
        self.temperature_end = epsilon_end    # 最小温度（例: 0.01）
        self.temperature_decay = epsilon_decay # 減衰率（例: 0.999）

    def _preprocess_state(self, state_img):
        """
        MazeEnv.get_image_observation() の出力 (4, rows, cols) を
        TransformerQNetwork の入力形式 (1, num_tokens, in_channels) に変換
        """
        # state_img: (4, rows, cols) の numpy 配列
        if isinstance(state_img, np.ndarray):
            state_img = torch.from_numpy(state_img).float()
        # (4, rows, cols) -> (rows*cols, 4) -> (1, num_tokens, in_channels)
        state_img = state_img.permute(1, 2, 0)  # (rows, cols, 4)
        state_img = state_img.reshape(-1, self.q_online.in_channels)  # (num_tokens, in_channels)
        state_img = state_img.unsqueeze(0)  # (1, num_tokens, in_channels)
        return state_img

    def select_action(self, state_img):
        """
        Q値のSoftmax確率分布に基づいて行動を選択（ボルツマン探索）
        state_img: MazeEnv.get_image_observation() の出力
        """
        self.q_online.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_state(state_img).to(self.device)
            q_values = self.q_online(state_tensor)  # (1, action_dim)
            
            # Q値を現在の温度でスケール（温度が低いほど最大値が強調される）
            scaled_q = q_values / max(self.temperature, 1e-8)
            
            # 確率分布に変換
            probs = torch.softmax(scaled_q, dim=1)  # (1, action_dim)
            
            # 確率分布から行動をサンプリング
            action = torch.multinomial(probs, num_samples=1).item()
            
        self.q_online.train()
        return action

    def store_experience(self, state_img, action, reward, next_state_img, done):
        """
        経験をバッファに保存
        """
        self.buffer.push(state_img, action, reward, next_state_img, done)

    def update(self):
        """
        経験再生バッファからミニバッチをサンプリングし、Qネットワークを更新（DDQN 風）
        """
        if len(self.buffer) < self.batch_size:
            return

        # バッファからサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # テンソルに変換
        states = torch.cat([self._preprocess_state(s) for s in states], dim=0).to(self.device)
        next_states = torch.cat([self._preprocess_state(s) for s in next_states], dim=0).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 現在の Q 値
        current_q = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ターゲット Q 値（DDQN 方式）
        with torch.no_grad():
            # オンライン側で次状態の最適行動を選択
            next_q_online = self.q_online(next_states)
            best_actions = next_q_online.argmax(dim=1)

            # ターゲット側でその行動の Q 値を評価
            next_q_target = self.q_target(next_states)
            max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 損失計算と更新
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # εの減衰
        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)

        # ターゲットネットワークの更新
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def train(self, num_episodes=1000, max_steps_per_episode=1000, maze_change=True, path=None):
        """
        学習ループ
        - maze_change=True: 各エピソードで迷路を再生成
        """
        episode_rewards = []

        for episode in range(num_episodes):
            # 迷路をリセット（必要に応じて再生成）
            state_pos = self.env.reset(maze_change=maze_change)
            state_img = self.env.get_image_observation()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state_img)
                next_state_pos, reward, done = self.env.step(action)
                next_state_img = self.env.get_image_observation()

                self.store_experience(state_img, action, reward, next_state_img, done)
                self.update()

                state_pos = next_state_pos
                state_img = next_state_img
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

            # ログ出力（任意）
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Temp: {self.temperature:.3f}")
            if episode % 50 == 0:
                print("save model")
                path_save = path if path else "maze_agent.pth"
                self.save_model(path_save)
        print("Training finished.")
        return episode_rewards

    def save_model(self, path):
        """
        モデルを保存
        """
        self.q_online.cpu()
        torch.save(self.q_online.state_dict(), path)
        self.q_online.to(self.device)

    def load_model(self, path):
        """
        モデルを読み込み
        """
        self.q_online.load_state_dict(torch.load(path))
        self.q_online.to(self.device)