from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import numpy as np
import gymnasium as gym
import pickle
import os
from stable_baselines3.common.evaluation import evaluate_policy

# Load expert demonstrations
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


# Create the environment
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# Initialize the GAIL model
learner = PPO("MlpPolicy", env, verbose=1)
gail = GAIL(
    venv=env,
    demo_batch_size=32,
    demonstrations=trajectories_input,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
# Train the GAIL model
gail.train(total_timesteps=10000)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))

# Save the trained model
learner.save(os.path.join(dir_current, "gail_cartpole_ppo"))
print("train complete and model saved.")