from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import gymnasium as gym
import pickle

# Create and train an expert model
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)


import numpy as np

trajectories = []

done = False

for _ in range(4000):  # Play for 1000 steps
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        next_obs, reward, done, info = env.step(action)
        trajectories.append((obs, action, reward, next_obs, done))
        obs = next_obs

# 保存
dir_current = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_current, "invader_expert.pickle"), mode="wb") as f:
    pickle.dump(trajectories, f)


