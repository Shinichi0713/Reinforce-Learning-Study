"""
overview:
学習により得られた方策(policy_estimatorが保持するニューラルネットワークの重み)を用いて
pybullet-gymのWalker2DPyBulletEnv-v0環境でhumanoidを歩かせる予測を行う

args:
weight_path: 学習済みのpolicy_estimatorのネットワークの重みのpath
(例: ../result/walker2d/episode_500000.h5)

output:
weight_pathと同じ階層に{weight_path}_movieディレクトリが生成され、結果のmp4が出力される
例:
├── episode_500000.h5
└── episode_500000_movie
   ├── openaigym.episode_batch.0.34390.stats.json
   ├── openaigym.manifest.0.34390.manifest.json
   ├── openaigym.video.0.34390.video000000.meta.json
   ├── openaigym.video.0.34390.video000000.mp4

usage:
python predict.py  {weight_path}
    例：predict.py  ../result/walker2d/episode_500000.h5
"""
import argparse
import os

import collections
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import pybullet as p
import pybulletgym.envs

from agent.policy_estimator import PolicyEstimator


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', help='learned model_weight path')
    args = parser.parse_args()
    weight_path = args.weight_path
    predict_dir = os.path.splitext(weight_path)[0]
    predict_movie_dir = os.path.join(predict_dir + '_movie')
    os.makedirs(predict_movie_dir, exist_ok=True)

    env = gym.make('Walker2DPyBulletEnv-v0')
    dim_state = env.env.observation_space.shape[0]
    dim_action = env.env.action_space.shape[0]
    env.render(mode='human')
    env = wrappers.Monitor(env, predict_movie_dir, force=True,
                           video_callable=(lambda ep: ep % 1 == 0))

    policy_estimator = PolicyEstimator(dim_state=dim_state,
                                       dim_action=dim_action)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _predict(sess, env, policy_estimator, weight_path)


def _predict(sess, env, policy_estimator, weight_path, num_episodes=5):
    policy_estimator.model.load_weights(weight_path)

    env.reset()
    torsoId = -1
    for i in range(p.getNumBodies()):
        print('body_info_{}'.format(p.getBodyInfo(i)))
        if p.getBodyInfo(i)[1].decode() == 'walker2d':
            torsoId = i

    Step = collections.namedtuple('Step', ['state', 'action', 'reward'])
    scores = []

    print('start_episodes...')
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode = []
        score = 0
        steps = 0
        while True:
            steps += 1
            action = policy_estimator.predict(sess, state)
            state_new, r, done, _ = env.step(action)
            score += r

            distance = 5
            yaw = 0
            humanPos = p.getLinkState(torsoId, 4)[0]
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
            still_open = env.render('human')
            if still_open is None:
                return

            episode.append(Step(state=state, action=action, reward=r))
            state = state_new
            if done:
                print('episode_{}, score_{}'.format(i_episode, score))
                scores.append(score)
                break
    print('{}episode avg_score_{}'.format(num_episodes, np.mean(scores)))


if __name__ == '__main__':
    predict()
