import gym
import pybulletgym.envs

env = gym.make('Walker2DPyBulletEnv-v0')
for _ in range(1):
    state = env.reset()
    while True:
        action = env.action_space.sample()
        state_new, r, done, info = env.step(action)
        print("reward: ", r)
        if done:
            print('episode done')
            break
