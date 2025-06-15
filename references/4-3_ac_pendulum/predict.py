"""
overview:
    Actor-Critic法による学習済みモデルを用いてPendulumの予測制御を行う

args:
    weight_path: 学習済みモデルのパス

output:
    {weight_path}/movie/ に num_batches 分の動画が出力される

usage-example:
    python3 predict.py result/20190211_021557/batch_400000.h5
"""
import argparse
import os
import collections

import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
from agent.actor import Actor


# 予測実行関数を定義
def predict(actions_list=(-1, 1)):
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', help='learned model_weight path')
    args = parser.parse_args()
    weight_path = args.weight_path
    predict_dir = os.path.splitext(weight_path)[0]
    predict_movie_dir = os.path.join(predict_dir, 'movie')
    os.makedirs(predict_movie_dir, exist_ok=True)

    # 環境モデルの指定
    env = gym.make("Pendulum-v0")

    # 環境モデルのパラメータ
    num_states = env.env.observation_space.shape[0]
    num_actions = len(actions_list)
    print('NUM_STATES_{}'.format(num_states))
    print('NUM_ACTIONS_{}'.format(num_actions))

    # predict_movie_dir にバッチごとに予測制御を mp4 形式で出力
    env = wrappers.Monitor(env, predict_movie_dir, force=True,
                           video_callable=(lambda ep: ep % 1 == 0))

    # actor のインスタンスを作成
    actor = Actor(num_states=num_states, actions_list=actions_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _predict(sess, env, actor, weight_path)


# バッチ予測制御の実行関数を定義
def _predict(sess, env, actor, weight_path, num_batches=10):
    # 学習済みモデルの重み係数をロード
    actor.model.load_weights(weight_path)

    env.reset()
    Step = collections.namedtuple("Step", ["state", "action", "reward"])
    last_10_score = np.zeros(10)

    print('start_batches')
    for i_batch in range(1, num_batches + 1):
        state = env.reset()
        batch = []
        score = 0
        steps = 0
        while True:
            steps += 1
            action = actor.predict(sess, state)
            state_new, reward, done, info = env.step([action])

            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1

            score += c_reward

            batch.append(Step(state=state, action=action, reward=c_reward))
            state = state_new
            if done:
                print('batch_{}, score_{}'.format(i_batch, score))
                break

        total_reward = sum(e.reward for e in batch)

        # 直近 10 バッチの報酬および報酬平均を記録
        last_10_score[i_batch % 10] = total_reward
        last_10_score_avg = sum(last_10_score) / min(i_batch, 10)
        if i_batch % 10 == 0:
            print('batch_{} score_{}  last 10_{}'
                  .format(i_batch, score, last_10_score_avg))


if __name__ == '__main__':
    predict()
