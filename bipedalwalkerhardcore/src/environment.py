
import gymnasium as gym

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


