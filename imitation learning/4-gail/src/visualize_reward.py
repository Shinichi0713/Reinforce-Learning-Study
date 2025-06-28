import numpy as np
import matplotlib.pyplot as plt
import torch
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.util.networks import RunningNorm
import gymnasium as gym
# 環境構築
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# 識別器の定義
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)


# 可視化する2次元（例：棒の角度と角速度）
theta_idx = 2   # 棒の角度
theta_dot_idx = 3  # 棒の角速度

theta_range = np.linspace(-0.2, 0.2, 50)
theta_dot_range = np.linspace(-2.0, 2.0, 50)

reward_map = np.zeros((len(theta_range), len(theta_dot_range)))

for i, theta in enumerate(theta_range):
    for j, theta_dot in enumerate(theta_dot_range):
        obs = np.array([[0.0, 0.0, theta, theta_dot]])
        next_obs = obs.copy()  # 必要なら状態遷移後の観測値をセット。ここでは同じ値で仮置き
        act = np.array([[0]])  # 行動0
        done = np.array([[0]]) # 終了でない場合

        # torchテンソルに変換
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        act_tensor = torch.tensor(act, dtype=torch.int64)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.float32)

        with torch.no_grad():
            reward = reward_net(obs_tensor, act_tensor, next_obs_tensor, done_tensor).cpu().numpy()[0]
        reward_map[i, j] = reward

plt.figure(figsize=(8, 6))
plt.imshow(
    reward_map.T,
    extent=[theta_range[0], theta_range[-1], theta_dot_range[0], theta_dot_range[-1]],
    origin='lower',
    aspect='auto'
)
plt.xlabel("theta (pole angle)")
plt.ylabel("theta_dot (pole angular velocity)")
plt.title("RewardNet output (action=0)")
plt.colorbar(label="reward")
plt.show()
