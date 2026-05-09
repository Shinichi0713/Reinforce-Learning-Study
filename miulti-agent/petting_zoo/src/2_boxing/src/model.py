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
    
def visualize_action_probs(agent, obs, agent_name="1P"):
    """
    特定の観測に対するエージェントの行動確率をグラフ化する
    """
    agent.eval()
    with torch.no_grad():
        # Actorネットワークを通してロジットを取得
        features = agent.actor_encoder(obs.unsqueeze(0))
        logits = agent.action_head(features)
        
        # ソフトマックス関数で確率(0.0~1.0)に変換
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # グラフ描画
    plt.figure(figsize=(12, 5))
    colors = ['skyblue' if "FIRE" not in name else 'salmon' for name in ACTION_MEANING]
    plt.bar(ACTION_MEANING, probs, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Probability")
    plt.title(f"Action Probability Distribution for {agent_name}")
    plt.ylim(0, 1.0) # 確率なので最大は1.0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




class MAPPORolloutBuffer:
    def __init__(self, buffer_size, obs_shape, joint_shape, device="cpu"):
        self.device = device
        self.buffer_size = buffer_size

        # 各エージェント(2人分)のデータを保持するテンソル
        # obs: (buffer_size, 2, 4, 84, 84)
        self.obs = torch.zeros((buffer_size, 2, *obs_shape), device=device)
        # joint_states: (buffer_size, 8, 84, 84) -> 集中クリティック用
        self.joint_states = torch.zeros((buffer_size, *joint_shape), device=device)
        
        self.actions = torch.zeros((buffer_size, 2), device=device)
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
