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
    def __init__(self, cnn_encoder, num_agents=4, hidden_size=64):
        super().__init__()
        self.cnn = cnn_encoder
        # グローバル状態の次元: num_agents * cnn_output_dim
        state_dim = num_agents * self.cnn.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, global_state):
        value = self.mlp(global_state)
        return value


def test_network_with_env():
    """
    修正版：PursuitWrapper環境でActor/Criticネットワークが正しく動作するかを確認するテスト。
    """
    env = PursuitWrapper(render_mode=None, max_cycles=10)
    obs_shape = (7, 7, 3)
    act_dim = env.action_dim
    num_agents = env.num_agents
    hidden_size = 64

    cnn_encoder = CNNEncoder(obs_shape)
    actor = Actor(cnn_encoder, act_dim, hidden_size)
    critic = Critic(cnn_encoder, num_agents, hidden_size)

    print(f"観測次元: {env.obs_dim}")
    print(f"グローバル状態次元（観測ベース）: {env.state_dim}")
    print(f"CNN出力次元: {cnn_encoder.output_dim}")
    print(f"Critic入力次元（CNNベース）: {num_agents * cnn_encoder.output_dim}")
    print(f"行動空間: {act_dim}")
    print(f"エージェント数: {num_agents}")
    print("--- テスト開始 ---")

    env.reset()
    step_count = 0
    max_steps = 10  # デバッグ用に短く

    for agent in env.env.agent_iter():
        if step_count >= max_steps:
            print(f"最大ステップ数({max_steps})に達したため終了")
            break
        step_count += 1

        obs_np = env.get_obs(agent)
        print(f"[Step {step_count}] Agent: {agent}")
        print(f"  obs_np shape: {obs_np.shape if obs_np is not None else None}")

        if obs_np is None:
            action = None
            log_prob = None
            value = None
        else:
            # NumPy → PyTorchテンソル（バッチ次元追加）
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).float()  # (1,147)
            print(f"  obs_tensor shape: {obs_tensor.shape}")

            # Actor: 行動とログ確率を計算
            dist = actor(obs_tensor)
            action_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(action_tensor)
            action = action_tensor.item()
            log_prob = log_prob_tensor.item()

            # Critic用のグローバル状態を構築（CNN特徴を結合）
            agent_features = []
            for a in env.possible_agents:
                if a in env.env.agents:
                    a_obs_np = env.get_obs(a)
                    if a_obs_np is not None:
                        a_obs_tensor = torch.from_numpy(a_obs_np).unsqueeze(0).float()
                        a_feat = cnn_encoder(a_obs_tensor)  # CNNでエンコード
                        agent_features.append(a_feat)
            if agent_features:
                global_state_tensor = torch.cat(agent_features, dim=-1)  # (1, num_agents * cnn_output_dim)
            else:
                global_state_tensor = None

            print(f"  global_state_tensor shape: {global_state_tensor.shape if global_state_tensor is not None else None}")

            if global_state_tensor is not None:
                # Critic: 状態価値を計算
                value_tensor = critic(global_state_tensor)
                value = value_tensor.item()
            else:
                value = None

            print(f"  action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

        reward, terminated, truncated, info = env.step(agent, action)
        print(f"  reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}")

        if terminated or truncated:
            print(f"エピソード終了（terminated: {terminated}, truncated: {truncated}）")
            break

    env.close()
    print("--- テスト終了 ---")


