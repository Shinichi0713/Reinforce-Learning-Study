import gymnasium as gym

# 環境コード
class Environment:
    def __init__(self, is_train=True):
        if is_train:
            self.env = gym.make("LunarLanderContinuous-v3")
        else:
            self.env = gym.make("LunarLanderContinuous-v3", render_mode="human")
        self.observation, self.info = self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.observation, self.info = self.env.reset()
        return self.observation

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


if __name__ == "__main__":
    # 環境の作成
    env = Environment(is_train=False)  # is_train=Trueで学習用、Falseでテスト用
    
    for step in range(500):  # 500ステップだけ実行
        # ランダムなアクションを選択
        action = env.env.action_space.sample()
        # アクションを実行
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation = env.reset()

    env.close()

