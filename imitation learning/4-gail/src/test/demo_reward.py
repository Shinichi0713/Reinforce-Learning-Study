from imitation.rewards.reward_nets import BasicRewardNet
import gym
import torch
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.util.networks import RunningNorm
import gymnasium as gym


# 環境構築
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# Define observation and action spaces
observation_space = env.observation_space
action_space = env.action_space

# Initialize the BasicRewardNet
reward_net = BasicRewardNet(
    observation_space=observation_space,
    action_space=action_space,
    use_action=True,  # Whether to include actions in the reward model
    use_next_state=False  # Whether to include next states in the reward model
)

# Example input tensors
obs = torch.tensor([env.observation_space.sample()], dtype=torch.float32)
acts = torch.tensor([env.action_space.sample()], dtype=torch.float32)

# Compute the reward
reward = reward_net(obs, acts, None, None)
print("Reward:", reward)
