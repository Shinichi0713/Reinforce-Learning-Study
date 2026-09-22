import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. 残差接続（Residual Block）付き CNN ブロック
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

# ---------------------------------------------------------
# 2. Hybrid CNN + Transformer Encoder モデル
# ---------------------------------------------------------
class CnnTransformerHybrid(nn.Module):
    def __init__(self, num_classes=10, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # [CNN エクストラクター]
        # [B, 3, 32, 32] -> [B, 32, 16, 16] -> [B, 64, 8, 8]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, d_model, kernel_size=3, stride=2, padding=1, bias=False), # 下挿 (16x16)
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            ResBlock(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False), # 下挿 (8x8)
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # 位置エンコーディング: 8x8 = 64 パッチ/トークン
        self.num_patches = 8 * 8
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # [Transformer エンコーダー]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [分類器]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 1. CNN 特徴抽出: [B, 3, 32, 32] -> [B, 64, 8, 8]
        feat = self.stem(x)
        
        # 2. トークン化（H, W 空間の平坦化 & 軸置換）: [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).permute(0, 2, 1)
        
        # 3. 位置エンコーディングの加算
        tokens = tokens + self.pos_embedding
        
        # 4. Transformer Encoder 処理: [B, 64, 64] -> [B, 64, 64]
        out_tokens = self.transformer(tokens)
        
        # 5. Global Average Pooling (シーケンス次元での平均化): [B, 64, 64] -> [B, 64]
        global_feat = out_tokens.mean(dim=1)
        
        # 6. クラス出力
        logits = self.head(global_feat)
        return logits
 