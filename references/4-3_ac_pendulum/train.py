"""
overview:
    Actor-Critic法によるPendulumの学習を行う

args:
    各種パラメータ設定値は、本コード中に明記される

output:
    RESULT_PATH に以下の要素が出力される
        - options.csv: バッチ学習のパラメータ設定
        - history.csv: バッチごとの（報酬, 損失関数, TD 誤差）
        - history.png: バッチごとの（報酬, 損失関数, TD 誤差）の可視化

usage-example:
    python3 train.py
"""
import csv
from datetime import datetime
import os
import collections

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from agent.actor import Actor
from agent.critic import Critic

# 学習パラメータ
train_config = {
    'num_batches': 40000,
    'gamma': 0.99,
    'interval': 10000,
    'batch_size': 50,
    'multi_step_td':
        True  # True：複数ステップ TD 誤差を計算、False：１ステップ TD 誤差を計算
}


def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)


# バッチ学習のパラメータ設定をファイル出力する関数
def write_options(csv_path, train_config):
    with open(csv_path, 'w') as f:
        fieldnames = ['Option Name', 'Option Value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [
            dict(zip(fieldnames, [k, v]))
            for k, v in train_config.items()
        ]
        writer.writerows(data)


# バッチ学習の metrics の推移を可視化する関数
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
    ax.set_xlabel('batch', fontsize=12)
    fig.suptitle(
        'window size is {0:d}.'.format(window_size))
    fig.savefig(png_path)
    plt.close(fig)


# 学習実行関数を定義
def train(train_config=train_config,
          actions_list=(-1, 1)):
    # 環境モデルの指定
    env = gym.make("Pendulum-v0")
    num_states = env.env.observation_space.shape[0]
    num_actions = len(actions_list)
    print('NUM_STATE_{}'.format(num_states))
    print('NUM_ACTIONS_{}'.format(num_actions))

    # actor と critic それぞれのインスタンスを作成
    actor = Actor(num_states=num_states,
                  actions_list=actions_list)
    critic = Critic(num_states=num_states)

    # 学習を実行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _train(sess, env, actor, critic, train_config,
               actions_list)


# バッチ TD 学習の実行関数
def _train(sess, env, actor, critic, train_config,
           actions_list):
    # 学習結果の出力先を指定
    result_dir = './result/{now_str}' \
                 .format(now_str=now_str(str_format='%Y%m%d_%H%M%S'))
    # 結果出力の保存先ディレクトリを作成
    os.makedirs(result_dir, exist_ok=True)
    print('result_dir_{}'.format(result_dir))

    # ログファイル名の指定
    csv_path = os.path.join(result_dir, 'history.csv')
    png_path = os.path.join(result_dir, 'history.png')
    opt_path = os.path.join(result_dir, 'options.csv')
    metrics = ['score', 'loss', 'loss_v']
    header = ['batch'] + metrics

    # バッチ学習の metrics を書き出す csv を作成
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # バッチ学習パラメータの設定値を csv に出力
    write_options(opt_path, train_config)

    # バッチ学習パラメータの読み出し
    num_batches = train_config['num_batches']
    batch_size = train_config['batch_size']
    gamma = train_config['gamma']
    interval = train_config['interval']
    multi_step_td = train_config['multi_step_td']

    Step = collections.namedtuple(
        "Step", ["state", "act_onehot", "reward"])
    last_100_score = np.zeros(100)

    print('start_batches...')
    for i_batch in range(1, num_batches + 1):
        state = env.reset()
        batch = []
        score = 0
        steps = 0
        while True:
            steps += 1
            action = actor.predict(sess, state)
            act_onehot = to_categorical(
                actions_list.index(action),
                len(actions_list))
            state_new, reward, done, info = \
                env.step([action])

            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1

            score += c_reward

            batch.append(
                Step(state=state,
                     act_onehot=act_onehot,
                     reward=c_reward))
            state = state_new

            if steps >= batch_size:
                break

        value_last = critic.predict(sess, state)[0][0]

        # １バッチ分のサンプリング終了後に TD 誤差を計算
        targets = []
        states = []
        act_onehots = []
        advantages = []
        target = value_last
        for t, step in reversed(list(enumerate(batch))):
            current_value = critic.predict(
                sess, step.state)[0][0]

            # １ステップ先の目標値、または複数ステップ先の目標値を計算
            if multi_step_td:
                target = step.reward + gamma * target
            else:
                target = step.reward + gamma * value_last
                value_last = current_value

            # アドバンテージ関数を１ステップ TD 誤差、
            # または複数ステップ TD 誤差として計算
            advantage = target - current_value
            targets.append([target])
            advantages.append([advantage])
            states.append(step.state)
            act_onehots.append(step.act_onehot)

        # Actor と Critic それぞれの損失関数を計算
        loss = actor.update(sess, states,
                            act_onehots,
                            advantages)
        loss_v = critic.update(sess, states, targets)

        # 直近 100 バッチの報酬および報酬平均を記録
        last_100_score[i_batch % 100] = score
        last_100_score_avg = sum(last_100_score) / min(
            i_batch, 100)

        batch_score_loss = [i_batch, score, loss, loss_v]

        # 学習のmetricを保存するcsvに結果の追記
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(batch_score_loss)

        # 100 ステップごとにログを書き出す
        if i_batch % 100 == 0:
            print(
                'batch_{} score_{} avg_loss_{} avg_td2_{} last100_score_{}'
                .format(i_batch, score, loss, loss_v,
                        last_100_score_avg))
            visualize_history(csv_path, png_path,
                              metrics)

        # 指定した間隔ごとに、actor の重み係数を保存
        if i_batch % interval == 0:
            model_path = os.path.join(
                result_dir,
                'batch_{}.h5'.format(i_batch))
            actor.model.save(model_path)


if __name__ == "__main__":
    train()
