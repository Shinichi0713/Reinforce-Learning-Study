
import gym
from collections import deque
import numpy as np


# 環境コード
class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("BipedalWalkerHardcore-v3")
        else:
            self.env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")
        self.observation, self.info = self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self):
        self.observation, self.info = self.env.reset()
        return self.observation

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    # 環境の状態数、行動数を返す
    def get_dimensions(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]


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

    def render(self):
        for _ in range(self._skip):
            out = self.env.render()
        return out

# 系列データへ変換するラッパー
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


if __name__ == "__main__":
    env = gym.make("BipedalWalkerHardcore-v3" , render_mode="human")
    env = MyWalkerWrapper(env)
    env = BoxToHistoryBox(env, h=8)

    obs = env.reset()
    print("Observation shape:", obs.shape)

    for _ in range(10):
        while True:
            if env.env.action_space.shape[0] == 1:
                action = env.env.action_space.sample()[0]  # ランダムな行動
            else:
                action = env.env.action_space.sample()  # ランダムな行動
            obs, total_reward, terminated, truncated, info = env.step(action)
            print("Step:", _, "Reward:", total_reward, "Done:", terminated)

            if terminated or truncated  :
                obs = env.reset()
                break
        print("Reset observation shape:", obs.shape)
        env.render()
    env.close()
