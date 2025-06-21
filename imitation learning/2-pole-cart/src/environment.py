
import gym
import numpy as np
import time, os


class Env():
    def __init__(self, is_train=False):
        if is_train:
            self.env = gym.make('CartPole-v1')
        else:
            self.env = gym.make('CartPole-v1', render_mode='human')
        self.reset()

    def reset(self):
        state = self.env.reset()
        return state, 0
    
    def step(self, action):
        state, reward, done, _, _ = self.env.step(action)
        return state, reward, done
    
    def render(self):
        self.env.render()
        time.sleep(0.01)


if __name__ == "__main__":
    env = Env()
    state, _ = env.reset()
    done = False
    while not done:
        action = env.env.action_space.sample()  # ランダムな行動を選択
        state, reward, done = env.step(action)
        env.render()
    env.env.close()