from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import numpy as np
import gymnasium as gym
import pickle
import os
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

# Load expert demonstrations
dir_current = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_current, "invader_expert.pickle"), "rb") as f:
    trajectories = pickle.load(f)
from imitation.data.types import Trajectory

# 各要素を取り出してリスト化
obs_list = []
acts_list = []
dones_list = []
infos_list = []

"""obs, action, reward, next_obs, done"""
for tup in trajectories:
    obs, act, rew, next_obs, done = tup
    obs_list.append(obs)
    acts_list.append(act)
    dones_list.append(bool(done[0]))
    infos_list.append({})  # 必要なら

# 最後のnext_obsを追加して、obsをN+1個に
obs_list.append(next_obs)

# 配列化
obs_array = np.array(obs_list)
acts_array = np.array(acts_list)
dones_array = np.array(dones_list)

# Trajectory作成
traj = Trajectory(obs=obs_array, acts=acts_array, infos=infos_list, terminal=dones_array)

# 1つのリストに
# trajectories = [traj]
trajectories = [traj]


# def ensure_array(x):
#     # すでに配列ならそのまま、スカラーなら長さ1配列に
#     if isinstance(x, np.ndarray):
#         return x if x.ndim > 0 else np.array([x])
#     else:
#         return np.array([x])

# def convert_to_trajectory(tup):
#     obs, acts, rews, next_obs, dones = tup
#     obs = ensure_array(obs)
#     acts = ensure_array(acts)
#     dones = ensure_array(dones)
#     infos = [{} for _ in range(len(acts))]
#     return Trajectory(obs=obs, acts=acts, infos=infos, terminal=dones)

# trajectories = [convert_to_trajectory(tup) for tup in trajectories]
# Create the environment
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
# SEED = 42
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=np.random.default_rng(SEED),
#     n_envs=8,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
# )

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# Initialize the GAIL model
gail = GAIL(
    venv=env,
    demo_batch_size=1024,
    demonstrations=trajectories,
    gen_algo=PPO("MlpPolicy", env, verbose=1),
    reward_net=reward_net,
)

# Train the GAIL model
gail.train(total_timesteps=10000)
