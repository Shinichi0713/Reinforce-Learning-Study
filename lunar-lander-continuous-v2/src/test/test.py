import gymnasium as gym

# 環境の作成
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
# 環境の初期化
observation, info = env.reset()

for step in range(500):  # 500ステップだけ実行
    # ランダムなアクションを選択
    action = env.action_space.sample()
    # アクションを実行
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

