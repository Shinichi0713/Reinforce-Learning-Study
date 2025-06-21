
import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent.agent_dqn import AgentDQN

def train():
    env = Env(is_train=True)
    agent = AgentDQN()
    num_episodes = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent(state)
            state, reward, done = env.step(action)
            agent.learn(state, action, reward, done)
        if episode % 100 == 0:
            print(f"Episode {episode}: {reward}")
    env.close()



if __name__ == "__main__":
    print("start dqn pole problem")
    is_train = True
