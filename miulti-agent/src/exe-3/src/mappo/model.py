import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from google.colab import drive
import os
drive.mount('/content/drive')

CHECKPOINT_DIR = "/content/drive/MyDrive/rl_exprolation_strage"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1. 2D RoPE (Rotary Position Embedding) モジュール ---
class RotaryPositionEmbedding2D(nn.Module):
    """
    2Dグリッド空間 (x, y) に対応したRoPE
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.half_dim = dim // 2
        # 周波数スケールの設定
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_emb(self, pos: torch.Tensor):
        # pos: (Batch, N)
        sinusoid_inp = torch.einsum("bn,f->bnf", pos, self.inv_freq)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        # [sin, sin, cos, cos] 形式のインターリーブ
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        return sin, cos

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        """
        q, k: (Batch, Num_Heads, N, Head_Dim)
        coords: (Batch, N, 2) -> (x, y) 座標
        """
        x_coords = coords[..., 0].float()
        y_coords = coords[..., 1].float()

        sin_x, cos_x = self._get_emb(x_coords) # (Batch, N, Head_Dim/2)
        sin_y, cos_y = self._get_emb(y_coords)

        # x軸とy軸のエンコーディングを結合
        sin = torch.cat([sin_x, sin_y], dim=-1).unsqueeze(1) # (Batch, 1, N, Head_Dim)
        cos = torch.cat([cos_x, cos_y], dim=-1).unsqueeze(1)

        # 回転行列の適用: q' = q * cos + rotate_half(q) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)


# --- 2. RoPE付き Multi-Head Attention ---
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionEmbedding2D(self.head_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        # x: (Batch, N, Embed_Dim), coords: (Batch, N, 2)
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPEの適用
        q, k = self.rope(q, k, coords)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v) # (B, Num_Heads, N, Head_Dim)

        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(context)


# --- 3. Transformer Block ---
class TransformerBlockWithRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.ffn(self.norm2(x))
        return x


# --- 4. 完全な RoPE-Transformer Agent (Actor-Critic) ---
class RoPETransformerAgent(nn.Module):
    def __init__(
        self, 
        grid_size: int = 15, 
        num_agents: int = 3, 
        action_space: int = 5,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # マップ特徴抽出用 CNN (15x15 -> Feature)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, embed_dim)
        )

        # エージェント情報用エンコーダ (ID + Position)
        self.agent_id_emb = nn.Embedding(num_agents, embed_dim)
        self.agent_pos_emb = nn.Linear(2, embed_dim)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlockWithRoPE(embed_dim, num_heads, ff_dim=embed_dim * 2)
            for _ in range(num_layers)
        ])

        # Actor & Critic Heads
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs_dict: dict, device="cpu"):
        """
        obs_dict: MultiSensorSearchEnv._get_obs() から得られる辞書表現
        """
        # 1. データのテンソル化
        # 全エージェントの位置を集約: (Batch=1, Num_Agents, 2)
        positions = [obs_dict[i]["position"] for i in range(self.num_agents)]
        coords = torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 共有マップのテンソル化: (Batch=1, 1, Grid_Size, Grid_Size)
        shared_map = torch.tensor(obs_dict[0]["map"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # 2. エンコーディング
        map_feat = self.map_encoder(shared_map).unsqueeze(1) # (1, 1, Embed_Dim)

        agent_ids = torch.arange(self.num_agents, device=device).unsqueeze(0)
        id_feats = self.agent_id_emb(agent_ids) # (1, Num_Agents, Embed_Dim)
        pos_feats = self.agent_pos_emb(coords / float(self.grid_size)) # 正規化

        # エージェントのトークン生成
        agent_tokens = id_feats + pos_feats # (1, Num_Agents, Embed_Dim)

        # 全トークンの結合 (Map Token + Agent Tokens)
        # Note: Map tokenの座標は中央 (grid_size/2, grid_size/2) として指定
        map_coord = torch.tensor([[[self.grid_size / 2.0, self.grid_size / 2.0]]], device=device)
        all_coords = torch.cat([map_coord, coords], dim=1) # (1, 1 + Num_Agents, 2)
        all_tokens = torch.cat([map_feat, agent_tokens], dim=1) # (1, 1 + Num_Agents, Embed_Dim)

        # 3. Transformer による相互作用の計算 (RoPE適用)
        x = all_tokens
        for layer in self.layers:
            x = layer(x, all_coords)

        # 4. エージェントトークンから Actor / Critic を計算
        agent_outputs = x[:, 1:, :] # マップトークンを除外 (1, Num_Agents, Embed_Dim)

        action_logits = self.actor_head(agent_outputs).squeeze(0) # (Num_Agents, Action_Space)
        values = self.critic_head(agent_outputs).squeeze(0)       # (Num_Agents, 1)

        return action_logits, values


# --- 動作確認テスト ---
if __name__ == "__main__":

    env = MultiSensorSearchEnv(size=15, num_agents=3)
    obs = env.reset()

    # モデルのインスタンス化
    model = RoPETransformerAgent(
        grid_size=15, 
        num_agents=3, 
        action_space=5, 
        embed_dim=64
    )

    # 推論実行
    logits, values = model(obs)
    
    # アクションのサンプリング
    probs = F.softmax(logits, dim=-1)
    dist = Categorical(probs)
    actions = dist.sample()

    actions_dict = {i: actions[i].item() for i in range(env.num_agents)}

    print("=== Transformer Agent Outputs ===")
    print(f"Action Logits Shape: {logits.shape}")  # (3, 5)
    print(f"Values Shape:        {values.shape}")  # (3, 1)
    print(f"Selected Actions:    {actions_dict}")