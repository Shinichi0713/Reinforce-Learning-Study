import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import gym

# ViTの設定（小型モデル、入力96x96x3）
vit_config = ViTConfig(
    image_size=96,
    num_channels=3,
    patch_size=16,
    num_hidden_layers=4,
    hidden_size=256,
    num_attention_heads=4,
    intermediate_size=512,
    qkv_bias=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

class ViTActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.vit = ViTModel(vit_config)
        vit_out_dim = vit_config.hidden_size

        # Actor head（CarRacingは連続行動: 3次元）
        self.actor = nn.Sequential(
            nn.Linear(vit_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 出力を[-1,1]に制限
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(vit_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, 3, 96, 96)
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # [CLS]トークン
        action = self.actor(features)
        value = self.critic(features)
        return action, value

# 使い方例
env = gym.make("CarRacing-v2")
model = ViTActorCritic(action_dim=3)

obs = env.reset()[0]  # gym>=0.26
obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0  # (1, 3, 96, 96)
with torch.no_grad():
    action, value = model(obs)
print("action:", action, "value:", value)
