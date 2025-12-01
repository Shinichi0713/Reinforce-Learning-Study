"""
overview:
    Double_DQNによる学習済みモデルを用いてPendulumの予測制御を行う

args:
    weight_path: 学習済みモデルのパス

output:
    {weight_path}/movie/ に num_episode 分の動画が出力される

usage:
    python predict.py  {weight_path}
        例：python predict.py result/pendulum/201902201257/episode_100.h5
"""
import argparse
import collections
import os

import gym
from gym import wrappers
import numpy as np

from agent.model import Qnetwork
from agent.policy import EpsilonGreedyPolicy


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', help='learned model_weight path')
    args = parser.parse_args()
    weight_path = args.weight_path
    max_step = 200  # 1エピソードの最大ステップ数
    actions_list = [-1, 1]  # 行動(action)の取りうる値のリスト
    num_episode = 20
    base_dir = os.path.dirname(weight_path)
    predict_movie_dir = os.path.join(base_dir, 'movie')

    os.makedirs(predict_movie_dir, exist_ok=True)

    env = gym.make("Pendulum-v0")
    env = wrappers.Monitor(env, predict_movie_dir, force=True,
                           video_callable=(lambda ep: ep % 1 == 0))
    dim_state = env.env.observation_space.shape[0]

    q_network = Qnetwork(dim_state, actions_list)
    q_network.main_network.load_weights(weight_path)
    policy = EpsilonGreedyPolicy(q_network, epsilon=0)

    exps = collections.namedtuple("exps", ["state", "action", "reward"])
    last_10_totalrewards = np.zeros(10)
    print('start_episodes')
    for episode_step in range(1, num_episode + 1):
        state = env.reset()
        episode_history = []
        score = 0
        step = 0
        while True:
            step += 1
            action, epsilon, q_values = policy.get_action(state, actions_list)
            next_state, reward, done, info = env.step([action])
            if reward > -1:
                reward = 1
            else:
                reward = -1
            score += reward
            episode_history.append(exps(state=state,
                                        action=action,
                                        reward=reward))
            state = next_state
            if step > max_step or done:
                print('episode_{}, score_{}'.format(episode_step, score))
                break

        total_reward = sum(e.reward for e in episode_history)
        last_10_totalrewards[episode_step % 10] = total_reward
        last_10_avg = sum(last_10_totalrewards) / \
            (episode_step if episode_step < 10 else 10)
        if episode_step % 10 == 0:
            print('episode_{} score_{}  last 10_{}'
                  .format(episode_step, score, last_10_avg))


if __name__ == '__main__':
    predict()
