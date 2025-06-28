from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import numpy as np
import gymnasium as gym
import pickle
import torch
import os
from stable_baselines3.common.evaluation import evaluate_policy

# エキスパートデータのロード
dir_current = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_current, "invader_expert.pickle"), "rb") as f:
    trajectories = pickle.load(f)
from imitation.data.types import Trajectory

# 各要素を取り出してリスト化
trajectories_input = []
"""obs, action, reward, next_obs, done"""
for tup in trajectories:
    obs, act, rew, next_obs, done = tup
    obs_input = np.concatenate([obs, next_obs], axis=0)
    traj = Trajectory(obs=obs_input, acts=act, infos=[{}], terminal=bool(done[0]))
    trajectories_input.append(traj)


# 環境構築
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# 識別器の定義
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# 生成器の定義
learner = PPO("MlpPolicy", env, verbose=1)

# GAILの定義
gail = GAIL(
    venv=env,
    demo_batch_size=32,
    demonstrations=trajectories_input,
    gen_algo=learner,
    reward_net=reward_net,
    log_dir=os.path.join(dir_current, "gail_cartpole"),
    allow_variable_horizon=True,
)

# GAILの学習前に、生成器の報酬を評価
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
# GAILの学習
gail.train(total_timesteps=10000)

# GAILの学習後に、生成器の報酬を評価
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))

# 行動生成器の保存
learner.save(os.path.join(dir_current, "gail_cartpole_ppo"))
# 識別器の保存
torch.save(reward_net.state_dict(), os.path.join(dir_current, "gail_cartpole_reward_net.pth"))
print("train complete and model saved.")
