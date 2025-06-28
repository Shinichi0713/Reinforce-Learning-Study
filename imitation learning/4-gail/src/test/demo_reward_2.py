import numpy as np
import matplotlib.pyplot as plt
from imitation.rewards.reward_nets import BasicRewardNet
import torch
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import os, time
from stable_baselines3.common.evaluation import evaluate_policy

# 環境を再生成し、モデルをロード
env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="human")])
dir_current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = PPO.load(os.path.join(dir_current, "gail_cartpole_ppo"), env=env)
# Example environment observation space
from gym.spaces import Box
obs_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
act_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

# Initialize BasicRewardNet
# 識別器の定義
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

obs = env.reset()
done = False

while not done:
    # 予測
    action, _ = model.predict(obs, deterministic=True)
    # 環境を進める
    obs_, reward, terminated, truncated = env.step(action)
    done = terminated[0] or truncated[0]
    # 報酬ネットで報酬を予測
    # obs, actionはnumpy配列なのでテンソルに変換
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    action_tensor = torch.as_tensor(action, dtype=torch.float32)
    # 入力shapeに注意（バッチ次元が必要な場合が多い）
    if len(obs_tensor.shape) == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    if len(action_tensor.shape) == 1:
        action_tensor = action_tensor.unsqueeze(0)
    predicted_reward = reward_net(obs_tensor, action_tensor, torch.as_tensor(obs_), done=torch.as_tensor(done['TimeLimit.truncated']))
    print(f"Action: {action}, EnvReward: {reward}, PredictedReward: {predicted_reward.item()}")

    obs = obs_
    time.sleep(0.02)
    env.render()
