from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import numpy as np
import gymnasium as gym
import pickle
import os, time
from stable_baselines3.common.evaluation import evaluate_policy



# 環境を再生成し、モデルをロード
env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="human")])
dir_current = os.path.dirname(os.path.abspath(__file__))
model = PPO.load(os.path.join(dir_current, "gail_cartpole_ppo"), env=env)

# DummyVecEnv から本体の環境を取り出す
env_core = env.envs[0]

obs = env.reset()
done = False

while not done:
    # 予測
    action, _ = model.predict(obs, deterministic=True)
    # 環境を進める
    obs, reward, terminated, truncated = env.step(action)
    done = terminated[0] or truncated[0]
    # 少し待つとレンダリングが見やすい
    time.sleep(0.02)
    env.render()
    done = done['TimeLimit.truncated']
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

# 終了後に閉じる
env_core.close()

