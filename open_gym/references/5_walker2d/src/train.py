"""
overview:
pybullet-gymのWalker2DPyBulletEnv-v0環境でhumanoidを歩かせるように学習を行う

args:
各種パラメータ設定値は、本コード中に明記される
学習が進まなくなる可能性があるので、変更は非推奨
    - result_dir:
        結果を出力するディレクトリのpath
    - num_episodes:
        学習の繰り返しエピソード数(default: 500000)
    - max_episode_steps:
        エピソードの最大ステップ数(default: 200)
    - gamma:
        割引率(default: 0.99)
    - model_save_interval:
        何エピソードごとにpolicy_estimatorのネットワークの重みを出力するか
        (default: 10000)

output:
スクリプト内で指定したresult_dirに以下のファイルが出力される
- episode_xxx.h5: xxxエピソード目の学習済みのpolicy_estimatorネットワークの重み
- history.csv: エピソードごとの以下の3つのメトリックを記録するcsv
    - score: 1エピソードで得られた報酬和
    - steps/episode: 1エピソードで倒れずに繰り返したステップ数
    - loss: policyestimatorのloss
- history.png: 上記3つのメトリックの推移を可視化した学習曲線

usage:
python train.py
"""
import collections
import csv
from datetime import datetime
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pybulletgym.envs

from agent.policy_estimator import PolicyEstimator
from agent.value_estimator import ValueEstimator

plt.style.use('seaborn-darkgrid')


def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)


# 学習のmetricの可視化を行う関数
def visualize_history(csv_path,
                      png_path,
                      metrics,
                      window_size=0.1):
    df = pd.read_csv(csv_path)
    if window_size < 1:
        window_size = max(int(len(df) * window_size), 1)
    else:
        window_size = int(window_size)

    rolled = df.rolling(window=window_size,
                        center=True).mean()
    fig = plt.figure(figsize=(12, 3 * len(metrics)))
    axes = fig.subplots(nrows=len(metrics), sharex=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        rolled[metric].plot(ax=ax,
                            title=metric,
                            grid=True,
                            legend=True)
        ax.get_legend().remove()
    ax.set_xlabel('episode', fontsize=12)
    fig.suptitle('Walker2DPyBulletEnv-v0')
    fig.savefig(png_path)
    plt.close(fig)


def train():
    env = gym.make('Walker2DPyBulletEnv-v0')
    dim_state = env.env.observation_space.shape[0]
    dim_action = env.env.action_space.shape[0]

    policy_estimator = PolicyEstimator(
        dim_state=dim_state, dim_action=dim_action)
    value_estimator = ValueEstimator(dim_state=dim_state)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _train(sess, env, policy_estimator,
               value_estimator)


def _train(sess, env, policy_estimator, value_estimator):
    # 結果を出力するディレクトリのpath
    result_dir = './result/walker2d/{now_str}' \
                 .format(now_str=now_str(str_format='%Y%m%d_%H%M%S'))
    os.makedirs(result_dir, exist_ok=True)
    print('result_dir_{}'.format(result_dir))

    num_episodes = 500000  # 学習の繰り返しエピソード数
    max_episode_steps = 200  # 1エピソードの最大ステップ数
    gamma = 0.99  # 割引率
    # 何エピソードごとにpolicy_estimatorのネットワークの重みをは出力するか
    model_save_interval = 10000

    # 学習のmetricを保存するcsv, 可視化するpngのpath
    csv_path = os.path.join(result_dir, 'history.csv')
    png_path = os.path.join(result_dir, 'history.png')
    metrics = ['steps/episode', 'score', 'loss']
    header = ['episode'] + metrics

    # 学習のmetricを保存するcsvの作成
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    Step = collections.namedtuple(
        'Step', ['state', 'action', 'reward'])
    last_100_score = np.zeros(100)
    last_100_steps = np.zeros(100)

    print('start_episodes...')
    # 数万episodeの繰り返し
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode = []
        score = 0
        steps = 0
        while True:
            # 確率的方策をもとに次に行うactionを取得
            steps += 1
            action = policy_estimator.predict(
                sess, state)
            state_new, r, done, _ = env.step(action)
            score += r

            episode.append(
                Step(state=state,
                     action=action,
                     reward=r))
            state = state_new  # ここまでが1step

            # 倒れる or Max stepで1episode終了
            if steps > max_episode_steps or done:
                break

        # 1episode終了後
        targets = []
        states = []
        actions = []
        advantages = []

        for t, step in enumerate(episode):
            # 割引報酬和 G_tの計算
            target = sum(
                gamma**i * t2.reward
                for i, t2 in enumerate(episode[t:]))
            # baseline = V(S_t), advantage = G_t - V(S_t)
            baseline_value = value_estimator.predict(
                sess, step.state)[0][0]
            advantage = target - baseline_value
            targets.append([target])
            advantages.append([advantage])
            states.append(step.state)
            actions.append(step.action)

        # policy_estimatorとvalue_estimatorの更新
        loss = policy_estimator.update(sess,
                                       states,
                                       actions,
                                       advantages)
        _ = value_estimator.update(sess, states, targets)

        last_100_steps[i_episode % 100] = steps
        last_100_score[i_episode % 100] = score

        episode_steps_score_loss = [
            i_episode, steps, score, loss
        ]
        # 学習のmetricを保存するcsvに結果の追記
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(episode_steps_score_loss)

        if i_episode % 100 == 0:
            last_100_steps_avg = sum(last_100_steps) \
                / (i_episode if i_episode < 100 else 100)
            last_100_score_avg = sum(last_100_score) \
                / (i_episode if i_episode < 100 else 100)
            print(
                'episode_{} last100_steps_avg_{} last100_score_avg_{}'
                .format(i_episode, last_100_steps_avg,
                        last_100_score_avg))
            visualize_history(csv_path, png_path,
                              metrics)

        if i_episode % model_save_interval == 0:
            model_path = os.path.join(
                result_dir,
                'episode_{}.h5'.format(i_episode))
            policy_estimator.model.save(model_path)


if __name__ == '__main__':
    train()
