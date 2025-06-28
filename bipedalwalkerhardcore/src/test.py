import gym

env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")              # GUI環境の開始(***)

for episode in range(20):
  observation = env.reset()                             # 環境の初期化
  for _ in range(100):
    env.render()                                        # レンダリング(画面の描画)
    action = env.action_space.sample()                  # 行動の決定
    observation, reward, done, terminate, info = env.step(action)  # 行動による次の状態の決定
    print("=" * 10)
    print("action=",action)
    print("observation=",observation)
    print("reward=",reward)
    print("done=",done)
    print("info=",info)

env.close()                                             # GUI環境の終了
