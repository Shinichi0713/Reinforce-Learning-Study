
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random, os
from archs.trsf_models import Actor, Critic
from torch.distributions import Normal
from env_wrappers import BoxToHistoryBox, MyWalkerWrapper

EPS = 0.003

def test_sac():
    env = gym.make('BipedalWalkerHardcore-v3', render_mode='human')
    env = MyWalkerWrapper(env, skip=2)
    env = BoxToHistoryBox(env, h=18)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(24, action_dim).to(device)
    dir_current = os.path.dirname(os.path.abspath(__file__))
    actor.load_state_dict(torch.load(f'{dir_current}/trsf_actor.pth', map_location=device), strict=False)
    actor.eval()

    for i in range(10):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor, explore=False)
            action = action.cpu().data.numpy()[0]
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            total_reward += reward
            state = next_state

        print(f"Episode {i+1}: Total Reward: {total_reward}")



if __name__ == "__main__":
    test_sac()
    # You can add more tests or functionality here as needed.