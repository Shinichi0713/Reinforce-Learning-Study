import gym
# import gym_ple  # noqa
import time
import numpy

import tensorflow as tf
from tensorflow.keras import backend
# from keras.backend import tensorflow_backend as backend

def main():
    env = gym.make("Breakout-v0")
    num_action = env.action_space.n
    print("num_action = " , num_action)
    episode_count = 200
    numpy.set_printoptions(threshold=numpy.inf)

    observation = env.reset()
    for e in range(episode_count):
        done = False

        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(e, "reward=",reward, ", done=",done, ", info=",info, ", action=",action)

            if done:
                observation = env.reset()
                break

            time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    main()

    backend.clear_session()