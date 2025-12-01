import gym
import pybulletgym.envs
from gym import wrappers
import pybullet as p
import os


predict_dir = './test'
os.makedirs(predict_dir, exist_ok=True)
env = gym.make('Walker2DPyBulletEnv-v0')
env.render(mode='human')
env = wrappers.Monitor(env, predict_dir, force=True,
                       video_callable=(lambda ep: ep % 1 == 0))
env.reset()
torsoId = -1
for i in range(p.getNumBodies()):
    print('body_info_{}'.format(p.getBodyInfo(i)))
    if p.getBodyInfo(i)[1].decode() == 'walker2d':
        torsoId = i

for _ in range(2):
    state = env.reset()
    while True:
        action = env.action_space.sample()
        state_new, r, done, info = env.step(action)
        print("reward: ", r)
        distance = 5
        yaw = 0
        humanPos = p.getLinkState(torsoId, 4)[0]
        p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
        still_open = env.render('human')

        if done:
            print('episode done')
            break
