import gym

env = gym.make('CartPole-v1', render_mode="human")
observation = env.reset()      # 初期化

for _ in range(1000):
    env.render()               # 画面に表示
    action = env.action_space.sample()  # ランダムに行動
    observation, reward, done, info, _ = env.step(action)  # 行動を実行
    if done:
        observation = env.reset()
env.close()