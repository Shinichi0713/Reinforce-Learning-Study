import torch
import torch.nn as nn

class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels=4, grid_size=5, d_model=64, nhead=4,
                 num_layers=2, action_dim=4, hidden_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # 1. トークン埋め込み
        self.embedding = nn.Linear(in_channels, d_model)

        # 2. CLSトークンの追加 (ベースとなる固定ベクトル)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 【採用手法2】エージェントの現在座標 (Row, Col) をベクトルに変換する線形層
        self.agent_pos_encoder = nn.Linear(2, d_model)

        # 【採用手法1】2次元位置エンコーディングのためのEmbedding層
        # 縦方向(Row)と横方向(Col)の位置情報をそれぞれ個別に学習
        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.col_embedding = nn.Embedding(grid_size, d_model)
        
        # CLSトークン用の位置埋め込み（空間情報を持たないため独立して用意）
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # 4. Transformer Encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 出力MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )

        # 最終出力層の初期化を小さくする
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)

    def forward(self, x, agent_pos):
        """
        Args:
            x: 迷路の環境状態テンソル (batch, in_channels, grid_size, grid_size)
            agent_pos: エージェントの現在座標テンソル (batch, 2) -> [[row, col], ...]
        """
        batch_size = x.shape[0]

        # (batch, in_channels, H, W) -> (batch, H*W, in_channels)
        x = x.flatten(2).transpose(1, 2)
        x = self.embedding(x)  # (batch, num_tokens, d_model)

        # --- 【採用手法2】CLSトークンにエージェントの文脈をブレンド ---
        # エージェントの座標 (row, col) を特徴量に変換
        agent_feat = self.agent_pos_encoder(agent_pos.float())  # (batch, d_model)
        
        # 固定のCLSトークンにエージェントの特徴量を足し合わせる
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        cls_tokens = cls_tokens + agent_feat.unsqueeze(1)       # (batch, 1, d_model)

        # 先頭に結合
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_tokens + 1, d_model)

        # --- 【採用手法1】2次元位置エンコーディングの動的生成 ---
        # 5x5のグリッド座標インデックスを生成
        coords = torch.arange(self.grid_size, device=x.device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")  # 各(grid_size, grid_size)
        
        # 縦横のEmbeddingを取得して加算 -> 2次元の空間情報を保持
        spatial_pos = self.row_embedding(grid_y) + self.col_embedding(grid_x)
        spatial_pos = spatial_pos.view(1, self.num_tokens, d_model)     # (1, num_tokens, d_model)

        # CLS用の位置埋め込みと結合して (1, num_tokens + 1, d_model) に整形
        pos_embedding = torch.cat([self.cls_pos_embedding, spatial_pos], dim=1)

        # 位置エンコーディングの加算
        x = x + pos_embedding

        # Transformer処理
        features = self.transformer(x)  # (batch, num_tokens + 1, d_model)

        # 先頭のCLSトークンの特徴量を抽出
        cls_feature = features[:, 0]  # (batch, d_model)

        # 行動Q値を出力
        q_values = self.mlp(cls_feature)
        return q_values