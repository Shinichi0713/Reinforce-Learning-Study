import re
import matplotlib.pyplot as plt

# 提示されたログデータ
log_data_relu = """
Episode 0: Reward = -92.89, Captures = 0, Avg Reward = -92.89, Avg Captures = 0.00
Actor Loss: -0.0148 | Critic Loss: 0.0817 | Avg Entropy: 1.6078
Episode 1: Reward = -150.94, Captures = 0, Avg Reward = -121.92, Avg Captures = 0.00
Actor Loss: -0.0162 | Critic Loss: 0.0661 | Avg Entropy: 1.6080
Episode 2: Reward = -134.85, Captures = 0, Avg Reward = -126.23, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0492 | Avg Entropy: 1.6080
Episode 3: Reward = -120.37, Captures = 0, Avg Reward = -124.77, Avg Captures = 0.00
Actor Loss: -0.0159 | Critic Loss: 0.0512 | Avg Entropy: 1.6079
Episode 4: Reward = -94.46, Captures = 0, Avg Reward = -118.70, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0488 | Avg Entropy: 1.6078
Episode 5: Reward = -125.08, Captures = 0, Avg Reward = -119.77, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0542 | Avg Entropy: 1.6078
Episode 6: Reward = -136.06, Captures = 0, Avg Reward = -122.10, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0366 | Avg Entropy: 1.6077
Episode 7: Reward = -127.98, Captures = 0, Avg Reward = -122.83, Avg Captures = 0.00
Actor Loss: -0.0171 | Critic Loss: 0.0428 | Avg Entropy: 1.6076
Episode 8: Reward = -121.48, Captures = 0, Avg Reward = -122.68, Avg Captures = 0.00
Actor Loss: -0.0171 | Critic Loss: 0.0488 | Avg Entropy: 1.6077
Episode 9: Reward = -114.44, Captures = 0, Avg Reward = -121.86, Avg Captures = 0.00
Actor Loss: -0.0177 | Critic Loss: 0.0409 | Avg Entropy: 1.6076
Episode 10: Reward = -153.76, Captures = 0, Avg Reward = -124.76, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0376 | Avg Entropy: 1.6076
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 11: Reward = -148.98, Captures = 0, Avg Reward = -126.78, Avg Captures = 0.00
Actor Loss: -0.0184 | Critic Loss: 0.0286 | Avg Entropy: 1.6076
Episode 12: Reward = -129.66, Captures = 0, Avg Reward = -127.00, Avg Captures = 0.00
Actor Loss: -0.0193 | Critic Loss: 0.0312 | Avg Entropy: 1.6072
Episode 13: Reward = -142.24, Captures = 0, Avg Reward = -128.09, Avg Captures = 0.00
Actor Loss: -0.0187 | Critic Loss: 0.0535 | Avg Entropy: 1.6074
Episode 14: Reward = -129.47, Captures = 0, Avg Reward = -128.18, Avg Captures = 0.00
Actor Loss: -0.0166 | Critic Loss: 0.0329 | Avg Entropy: 1.6073
Episode 15: Reward = -100.46, Captures = 0, Avg Reward = -126.45, Avg Captures = 0.00
Actor Loss: -0.0189 | Critic Loss: 0.0773 | Avg Entropy: 1.6071
Episode 16: Reward = -136.73, Captures = 0, Avg Reward = -127.05, Avg Captures = 0.00
Actor Loss: -0.0165 | Critic Loss: 0.0393 | Avg Entropy: 1.6072
Episode 17: Reward = -89.46, Captures = 0, Avg Reward = -124.96, Avg Captures = 0.00
Actor Loss: -0.0133 | Critic Loss: 0.0481 | Avg Entropy: 1.6072
Episode 18: Reward = -138.63, Captures = 0, Avg Reward = -125.68, Avg Captures = 0.00
Actor Loss: -0.0195 | Critic Loss: 0.0424 | Avg Entropy: 1.6072
Episode 19: Reward = -151.93, Captures = 0, Avg Reward = -126.99, Avg Captures = 0.00
Actor Loss: -0.0188 | Critic Loss: 0.0268 | Avg Entropy: 1.6073
Episode 20: Reward = -144.71, Captures = 0, Avg Reward = -127.84, Avg Captures = 0.00
Actor Loss: -0.0161 | Critic Loss: 0.0269 | Avg Entropy: 1.6073
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 21: Reward = -101.36, Captures = 0, Avg Reward = -126.63, Avg Captures = 0.00
Actor Loss: -0.0203 | Critic Loss: 0.0623 | Avg Entropy: 1.6071
Episode 22: Reward = -163.56, Captures = 0, Avg Reward = -128.24, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0323 | Avg Entropy: 1.6071
Episode 23: Reward = -82.30, Captures = 0, Avg Reward = -126.33, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0564 | Avg Entropy: 1.6071
Episode 24: Reward = -127.81, Captures = 0, Avg Reward = -126.39, Avg Captures = 0.00
Actor Loss: -0.0168 | Critic Loss: 0.0369 | Avg Entropy: 1.6071
Episode 25: Reward = -94.26, Captures = 0, Avg Reward = -125.15, Avg Captures = 0.00
Actor Loss: -0.0211 | Critic Loss: 0.0724 | Avg Entropy: 1.6069
Episode 26: Reward = -97.42, Captures = 0, Avg Reward = -124.12, Avg Captures = 0.00
Actor Loss: -0.0171 | Critic Loss: 0.0566 | Avg Entropy: 1.6070
Episode 27: Reward = -137.14, Captures = 0, Avg Reward = -124.59, Avg Captures = 0.00
Actor Loss: -0.0185 | Critic Loss: 0.0371 | Avg Entropy: 1.6071
Episode 28: Reward = -148.75, Captures = 0, Avg Reward = -125.42, Avg Captures = 0.00
Actor Loss: -0.0182 | Critic Loss: 0.0326 | Avg Entropy: 1.6070
Episode 29: Reward = -149.28, Captures = 0, Avg Reward = -126.22, Avg Captures = 0.00
Actor Loss: -0.0184 | Critic Loss: 0.0421 | Avg Entropy: 1.6070
Episode 30: Reward = -83.64, Captures = 0, Avg Reward = -124.84, Avg Captures = 0.00
Actor Loss: -0.0200 | Critic Loss: 0.0724 | Avg Entropy: 1.6070
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 31: Reward = -114.05, Captures = 0, Avg Reward = -124.51, Avg Captures = 0.00
Actor Loss: -0.0180 | Critic Loss: 0.0621 | Avg Entropy: 1.6069
Episode 32: Reward = -128.93, Captures = 0, Avg Reward = -124.64, Avg Captures = 0.00
Actor Loss: -0.0139 | Critic Loss: 0.0330 | Avg Entropy: 1.6071
Episode 33: Reward = -156.58, Captures = 0, Avg Reward = -125.58, Avg Captures = 0.00
Actor Loss: -0.0150 | Critic Loss: 0.0285 | Avg Entropy: 1.6073
Episode 34: Reward = -169.56, Captures = 0, Avg Reward = -126.84, Avg Captures = 0.00
Actor Loss: -0.0178 | Critic Loss: 0.0432 | Avg Entropy: 1.6072
Episode 35: Reward = -125.43, Captures = 0, Avg Reward = -126.80, Avg Captures = 0.00
Actor Loss: -0.0184 | Critic Loss: 0.0460 | Avg Entropy: 1.6072
Episode 36: Reward = -164.68, Captures = 0, Avg Reward = -127.82, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.0256 | Avg Entropy: 1.6072
Episode 37: Reward = -75.59, Captures = 0, Avg Reward = -126.45, Avg Captures = 0.00
Actor Loss: -0.0174 | Critic Loss: 0.0790 | Avg Entropy: 1.6072
Episode 38: Reward = -133.58, Captures = 0, Avg Reward = -126.63, Avg Captures = 0.00
Actor Loss: -0.0151 | Critic Loss: 0.0371 | Avg Entropy: 1.6071
Episode 39: Reward = -158.56, Captures = 0, Avg Reward = -127.43, Avg Captures = 0.00
Actor Loss: -0.0178 | Critic Loss: 0.0324 | Avg Entropy: 1.6072
Episode 40: Reward = -90.19, Captures = 0, Avg Reward = -126.52, Avg Captures = 0.00
Actor Loss: -0.0227 | Critic Loss: 0.0669 | Avg Entropy: 1.6073
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 41: Reward = -134.75, Captures = 0, Avg Reward = -126.71, Avg Captures = 0.00
Actor Loss: -0.0174 | Critic Loss: 0.0390 | Avg Entropy: 1.6073
Episode 42: Reward = -121.44, Captures = 0, Avg Reward = -126.59, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0413 | Avg Entropy: 1.6073
Episode 43: Reward = -132.99, Captures = 0, Avg Reward = -126.74, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0317 | Avg Entropy: 1.6072
Episode 44: Reward = -143.59, Captures = 0, Avg Reward = -127.11, Avg Captures = 0.00
Actor Loss: -0.0153 | Critic Loss: 0.0296 | Avg Entropy: 1.6072
Episode 45: Reward = -136.20, Captures = 0, Avg Reward = -127.31, Avg Captures = 0.00
Actor Loss: -0.0198 | Critic Loss: 0.0421 | Avg Entropy: 1.6072
Episode 46: Reward = -129.43, Captures = 0, Avg Reward = -127.35, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0356 | Avg Entropy: 1.6071
Episode 47: Reward = -139.89, Captures = 0, Avg Reward = -127.62, Avg Captures = 0.00
Actor Loss: -0.0180 | Critic Loss: 0.0427 | Avg Entropy: 1.6071
Episode 48: Reward = -127.79, Captures = 0, Avg Reward = -127.62, Avg Captures = 0.00
Actor Loss: -0.0151 | Critic Loss: 0.0429 | Avg Entropy: 1.6072
Episode 49: Reward = -152.00, Captures = 0, Avg Reward = -128.11, Avg Captures = 0.00
Actor Loss: -0.0159 | Critic Loss: 0.0400 | Avg Entropy: 1.6070
Episode 50: Reward = -149.53, Captures = 0, Avg Reward = -128.53, Avg Captures = 0.00
Actor Loss: -0.0137 | Critic Loss: 0.0313 | Avg Entropy: 1.6070
"""

log_data_gelu = """
Episode 0: Reward = -126.81, Captures = 0, Avg Reward = -126.81, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.1027 | Avg Entropy: 1.6085
Episode 1: Reward = -96.68, Captures = 0, Avg Reward = -111.75, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.1005 | Avg Entropy: 1.6085
Episode 2: Reward = -125.40, Captures = 0, Avg Reward = -116.30, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0900 | Avg Entropy: 1.6085
Episode 3: Reward = -158.32, Captures = 0, Avg Reward = -126.80, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.1070 | Avg Entropy: 1.6085
Episode 4: Reward = -95.05, Captures = 0, Avg Reward = -120.45, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.1072 | Avg Entropy: 1.6086
Episode 5: Reward = -142.06, Captures = 0, Avg Reward = -124.05, Avg Captures = 0.00
Actor Loss: -0.0152 | Critic Loss: 0.0926 | Avg Entropy: 1.6085
Episode 6: Reward = -140.07, Captures = 0, Avg Reward = -126.34, Avg Captures = 0.00
Actor Loss: -0.0163 | Critic Loss: 0.0854 | Avg Entropy: 1.6086
Episode 7: Reward = -121.78, Captures = 0, Avg Reward = -125.77, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0947 | Avg Entropy: 1.6086
Episode 8: Reward = -149.45, Captures = 0, Avg Reward = -128.40, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0880 | Avg Entropy: 1.6086
Episode 9: Reward = -116.19, Captures = 0, Avg Reward = -127.18, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0961 | Avg Entropy: 1.6086
Episode 10: Reward = -148.00, Captures = 0, Avg Reward = -129.07, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0856 | Avg Entropy: 1.6086
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 11: Reward = -133.20, Captures = 0, Avg Reward = -129.42, Avg Captures = 0.00
Actor Loss: -0.0155 | Critic Loss: 0.0845 | Avg Entropy: 1.6086
Episode 12: Reward = -133.22, Captures = 0, Avg Reward = -129.71, Avg Captures = 0.00
Actor Loss: -0.0152 | Critic Loss: 0.0966 | Avg Entropy: 1.6086
Episode 13: Reward = -72.73, Captures = 0, Avg Reward = -125.64, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0961 | Avg Entropy: 1.6086
Episode 14: Reward = -139.48, Captures = 0, Avg Reward = -126.56, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0665 | Avg Entropy: 1.6086
Episode 15: Reward = -143.87, Captures = 0, Avg Reward = -127.64, Avg Captures = 0.00
Actor Loss: -0.0168 | Critic Loss: 0.0616 | Avg Entropy: 1.6086
Episode 16: Reward = -116.86, Captures = 0, Avg Reward = -127.01, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0652 | Avg Entropy: 1.6086
Episode 17: Reward = -138.34, Captures = 0, Avg Reward = -127.64, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.0519 | Avg Entropy: 1.6086
Episode 18: Reward = -118.95, Captures = 0, Avg Reward = -127.18, Avg Captures = 0.00
Actor Loss: -0.0165 | Critic Loss: 0.0555 | Avg Entropy: 1.6086
Episode 19: Reward = -117.86, Captures = 0, Avg Reward = -126.72, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0448 | Avg Entropy: 1.6086
Episode 20: Reward = -148.68, Captures = 0, Avg Reward = -127.76, Avg Captures = 0.00
Actor Loss: -0.0160 | Critic Loss: 0.0433 | Avg Entropy: 1.6086
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 21: Reward = -111.50, Captures = 0, Avg Reward = -127.02, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0430 | Avg Entropy: 1.6086
Episode 22: Reward = -141.88, Captures = 0, Avg Reward = -127.67, Avg Captures = 0.00
Actor Loss: -0.0159 | Critic Loss: 0.0384 | Avg Entropy: 1.6086
Episode 23: Reward = -141.98, Captures = 0, Avg Reward = -128.27, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0294 | Avg Entropy: 1.6086
Episode 24: Reward = -136.10, Captures = 0, Avg Reward = -128.58, Avg Captures = 0.00
Actor Loss: -0.0160 | Critic Loss: 0.0316 | Avg Entropy: 1.6086
Episode 25: Reward = -113.64, Captures = 0, Avg Reward = -128.00, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.0453 | Avg Entropy: 1.6086
Episode 26: Reward = -69.11, Captures = 0, Avg Reward = -125.82, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0737 | Avg Entropy: 1.6086
Episode 27: Reward = -125.86, Avg Reward = -125.82, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.0458 | Avg Entropy: 1.6086
Episode 28: Reward = -114.83, Captures = 0, Avg Reward = -125.45, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0589 | Avg Entropy: 1.6085
Episode 29: Reward = -141.32, Captures = 0, Avg Reward = -125.97, Avg Captures = 0.00
Actor Loss: -0.0189 | Critic Loss: 0.0324 | Avg Entropy: 1.6085
Episode 30: Reward = -139.23, Captures = 0, Avg Reward = -126.40, Avg Captures = 0.00
Actor Loss: -0.0166 | Critic Loss: 0.0327 | Avg Entropy: 1.6085
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 31: Reward = -176.79, Captures = 0, Avg Reward = -127.98, Avg Captures = 0.00
Actor Loss: -0.0150 | Critic Loss: 0.0383 | Avg Entropy: 1.6084
Episode 32: Reward = -121.66, Captures = 0, Avg Reward = -127.79, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0380 | Avg Entropy: 1.6084
Episode 33: Reward = -118.52, Captures = 0, Avg Reward = -127.51, Avg Captures = 0.00
Actor Loss: -0.0176 | Critic Loss: 0.0560 | Avg Entropy: 1.6083
Episode 34: Reward = -117.43, Captures = 0, Avg Reward = -127.23, Avg Captures = 0.00
Actor Loss: -0.0206 | Critic Loss: 0.0485 | Avg Entropy: 1.6083
Episode 35: Reward = -128.28, Captures = 0, Avg Reward = -127.25, Avg Captures = 0.00
Actor Loss: -0.0182 | Critic Loss: 0.0354 | Avg Entropy: 1.6082
Episode 36: Reward = -106.43, Captures = 0, Avg Reward = -126.69, Avg Captures = 0.00
Actor Loss: -0.0201 | Critic Loss: 0.0605 | Avg Entropy: 1.6080
Episode 37: Reward = -77.97, Captures = 0, Avg Reward = -125.41, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0656 | Avg Entropy: 1.6080
Episode 38: Reward = -139.04, Captures = 0, Avg Reward = -125.76, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0356 | Avg Entropy: 1.6080
Episode 39: Reward = -115.49, Captures = 0, Avg Reward = -125.50, Avg Captures = 0.00
Actor Loss: -0.0201 | Critic Loss: 0.0476 | Avg Entropy: 1.6078
Episode 40: Reward = -109.83, Captures = 0, Avg Reward = -125.12, Avg Captures = 0.00
Actor Loss: -0.0164 | Critic Loss: 0.0586 | Avg Entropy: 1.6078
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 41: Reward = -118.95, Captures = 0, Avg Reward = -124.97, Avg Captures = 0.00
Actor Loss: -0.0183 | Critic Loss: 0.0338 | Avg Entropy: 1.6076
Episode 42: Reward = -71.75, Captures = 0, Avg Reward = -123.74, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.0718 | Avg Entropy: 1.6079
Episode 43: Reward = -149.73, Captures = 0, Avg Reward = -124.33, Avg Captures = 0.00
Actor Loss: -0.0191 | Critic Loss: 0.0313 | Avg Entropy: 1.6079
Episode 44: Reward = -141.59, Captures = 0, Avg Reward = -124.71, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0321 | Avg Entropy: 1.6080
Episode 45: Reward = -137.33, Captures = 0, Avg Reward = -124.98, Avg Captures = 0.00
Actor Loss: -0.0177 | Critic Loss: 0.0346 | Avg Entropy: 1.6080
Episode 46: Reward = -136.45, Captures = 0, Avg Reward = -125.23, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.0393 | Avg Entropy: 1.6081
Episode 47: Reward = -136.99, Captures = 0, Avg Reward = -125.47, Avg Captures = 0.00
Actor Loss: -0.0190 | Critic Loss: 0.0306 | Avg Entropy: 1.6080
Episode 48: Reward = -143.60, Captures = 0, Avg Reward = -125.84, Avg Captures = 0.00
Actor Loss: -0.0173 | Critic Loss: 0.0284 | Avg Entropy: 1.6079
Episode 49: Reward = -121.92, Captures = 0, Avg Reward = -125.76, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0450 | Avg Entropy: 1.6080
Episode 50: Reward = -111.40, Captures = 0, Avg Reward = -125.48, Avg Captures = 0.00
Actor Loss: -0.0196 | Critic Loss: 0.0332 | Avg Entropy: 1.6079
"""

def parse_logs(log_text):
    """ログからActor LossとCritic Lossを取り出す関数"""
    actor_losses = []
    critic_losses = []
    
    # 正規表現で数値を抽出
    pattern = r"Actor Loss:\s*([-\d.]+)\s*\|\s*Critic Loss:\s*([\d.]+)"
    matches = re.findall(pattern, log_text)
    
    for actor, critic in matches:
        actor_losses.append(float(actor))
        critic_losses.append(float(critic))
        
    return actor_losses, critic_losses

# データのパース
relu_actor, relu_critic = parse_logs(log_data_relu)
gelu_actor, gelu_critic = parse_logs(log_data_gelu)
episodes = list(range(len(relu_actor)))

# グラフ描画の設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Actor Loss のプロット
ax1.plot(episodes, relu_actor, label='ReLU', color='blue', alpha=0.8)
ax1.plot(episodes, gelu_actor, label='GELU', color='orange', alpha=0.8)
ax1.set_title('Actor Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

# 2. Critic Loss のプロット
ax2.plot(episodes, relu_critic, label='ReLU', color='blue', alpha=0.8)
ax2.plot(episodes, gelu_critic, label='GELU', color='orange', alpha=0.8)
ax2.set_title('Critic Loss Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()