import gym
from collections import deque
import numpy as np

# https://alexandervandekleut.github.io/gym-wrappers/
#env = gym.make('BipedalWalker-v3')

class BoxToHistoryBox(gym.ObservationWrapper):
    '''
    This wrapper converts the environment which returns last h observations.
    First h observations are converted such that first states are same.
    '''
    def __init__(self, env, h=8):
        super().__init__(env)
        self.h = h
        self.obs_memory = deque(maxlen=self.h)
        shape = (h,) + self.observation_space.shape
        low = np.repeat(np.expand_dims(self.observation_space.low, 0), h, axis=0)
        high = np.repeat(np.expand_dims(self.observation_space.high, 0), h, axis=0)    
        self.observation_space = gym.spaces.Box(low, high, shape)

    def add_to_memory(self, obs):
        self.obs_memory.append(np.expand_dims(obs, axis=0))

    def observation(self, obs):
        self.add_to_memory(obs)
        return np.concatenate(self.obs_memory)

    def reset(self):
        reset_state = self.env.reset()[0]
        for i in range(self.h-1):
            self.add_to_memory(reset_state)
        return self.observation(reset_state)

class MyWalkerWrapper(gym.Wrapper):
    '''
    This is custom wrapper for BipedalWalker-v3 and BipedalWalkerHardcore-v3. 
    Rewards for failure is decreased to make agent brave for exploration and 
    time frequency of dynamic is lowered by skipping two frames.
    '''
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip
        self._max_episode_steps = 750
        
    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        for i in range(self._skip):
            obs, reward, t, tr, info = self.env.step(action)
            # 新APIならt, trが入ってくる
            terminated = terminated or t
            truncated = truncated or tr
            done = terminated or truncated
            if self.env.game_over:
                reward = -10.0
                info["dead"] = True
            else:
                info["dead"] = False
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        return obs, total_reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        for _ in range(self._skip):
            out = self.env.render(mode=mode)
        return out
