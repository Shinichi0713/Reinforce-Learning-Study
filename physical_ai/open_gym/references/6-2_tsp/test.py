"""
overview:
    学習済みモデルを用いた巡回セールスマン問題の解予測

args:
    各種パラメータ規定値は、本コード中に明記される
        - log_dir: 下記要素の出力先
        - model_path: 予測用学習済みモデルファイル
        - n_episodes: 予測エピソード数

output:
    上記log_dirに以下の要素を出力する
        - list_results.pkl: 予測結果のリスト

usage-example:
    python3 train.py --log_dir=./result \
    --model_path=./results/model.099800-0.394-2.65370.ckpt \
    --n_episodes=5000
"""
import os
import argparse
import time
import pickle

import numpy as np
import tensorflow as tf

from agent.actor_critic_agent import ActorCriticAgent
from gym_env.tsp_env import TSPEnv


class ObjectLogger(object):

    def __init__(self, log_dir):
        self.object_path = os.path.join(log_dir, 'list_results.pkl')

    def object_save(self, log_obj):
        with open(self.object_path, mode='wb') as ofs:
            pickle.dump(log_obj, ofs)


def get_args():
    # arg parserの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./result',
                        help='log directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to model ckpt file')
    parser.add_argument('--n_episodes', type=int, default=5000,
                        help='# of episodes for test')

    return parser.parse_args()


def test():
    # argsの取得
    args = get_args()
    log_dir = args.log_dir
    model_path = args.model_path
    n_episodes = args.n_episodes
    n_steps = 1

    os.makedirs(args.log_dir, exist_ok=True)

    # --- PRE-PROCESS ---
    # tf sessionのスタート
    sess = tf.Session()

    # インスタンスの作成
    env = TSPEnv(train_flg=False)
    agent = ActorCriticAgent()
    logger = ObjectLogger(log_dir)

    # ネットワーク変数の初期化
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # 学習済みモデルからネットワーク変数をrestore
    agent.restore_graph(sess, model_path)

    # --- TEST MAIN ---
    # monitoring metrics用の変数
    list_losses, list_rewards = [], []
    list_inputs, list_tours = [], []
    start_time = time.time()

    # episodeのループ
    for i_episode in range(n_episodes):

        state = env.reset()
        data = env.get_data()

        # stepのループ
        for i_step in range(n_steps):

            # loss推定（agent.predict、env.stepの一括実行に相当）
            losses, rewards, model_prds = agent.predict_loss(sess, state)
            loss, vloss, aloss = losses
            reward, tour_dist = rewards
            _, tour, _ = model_prds

            list_losses.append([loss, vloss, aloss])
            list_rewards.append(reward)
            list_inputs.append(data)
            list_tours.append(tour)

        # --- POST-PROCESS (EPISODE) ---
        i_episode += 1
        if not i_episode % 50:

            # monitoring metricsの算出
            duration = time.time() - start_time
            avg_loss, avg_vloss, avg_aloss = np.mean(list_losses, axis=0)
            avg_reward = np.mean(list_rewards)

            # monitoringのリセット
            start_time = time.time()

            # print
            log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode, n_episodes)
            log_str += ' - Time: {0:3.2f}'.format(duration)
            log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
            log_str += ' - Avg_Loss: {0:3.5f}'.format(avg_loss)
            log_str += ' - Avg_VLoss: {0:3.5f}'.format(avg_vloss)
            log_str += ' - Avg_ALoss: {0:3.5f}'.format(avg_aloss)
            print(log_str)

    # --- POST-PROCESS (TESTING) ---
    # 予測結果リストのオブジェクトの出力
    log_obj = [list_inputs, list_tours, list_rewards]
    logger.object_save(log_obj)


if __name__ == '__main__':
    test()
