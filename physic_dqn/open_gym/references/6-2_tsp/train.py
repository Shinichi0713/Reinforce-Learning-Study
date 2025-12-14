"""
overview:
    Actor-Critic法による巡回セールスマン問題の解検索

args:
    各種パラメータ規定値は、本コード中に明記される
        - log_dir: 下記要素の出力先
        - n_episodes: 学習エピソード数

output:
    上記log_dirに以下の要素を出力する
        - checkpoint: チェックポイントファイル
        - model.xxx-xxx.ckpt: チェックポイント毎のモデルファイル
        - reward_log.csv: 報酬、損失など監視指標のログ

usage-example:
    python3 train.py --log_dir=./result \
    --n_episodes=60000
"""
import os
import argparse
import time
import csv

import numpy as np
import tensorflow as tf

from agent.actor_critic_agent import ActorCriticAgent
from gym_env.tsp_env import TSPEnv


class HistoryLogger(object):

    def __init__(self, log_dir):
        self.history_path = os.path.join(log_dir, 'reward_log.csv')

    def set_history_header(self, log_header):
        with open(self.history_path, mode='w') as ofs:
            writer = csv.writer(ofs)
            writer.writerow(log_header)

    def history_save(self, log_list):
        with open(self.history_path, mode='a') as ofs:
            writer = csv.writer(ofs)
            writer.writerow(log_list)


def get_args():
    # arg parserの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./result',
                        help='log directory')
    parser.add_argument('--n_episodes', type=int, default=60000,
                        help='# of episodes for train')

    return parser.parse_args()


def train():
    # argsの取得
    args = get_args()
    log_dir = args.log_dir
    n_episodes = args.n_episodes
    n_steps = 1

    os.makedirs(args.log_dir, exist_ok=True)

    # --- PRE-PROCESS ---
    # tf sessionのスタート
    sess = tf.Session()

    # インスタンスの作成
    env = TSPEnv(train_flg=True)
    agent = ActorCriticAgent()
    logger = HistoryLogger(log_dir)

    # ネットワーク変数の初期化
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # history loggingのヘッダ定義
    _header = ['episode', 'avg_reward', 'avg_loss', 'avg_vloss', 'avg_aloss']
    logger.set_history_header(_header)

    # --- TRAIN MAIN ---
    # monotoring metrics用の変数
    min_metric = 0.0
    list_losses, list_rewards = [], []
    start_time = time.time()

    # episodeのループ
    for i_episode in range(n_episodes):

        state = env.reset()

        # stepのループ
        for i_step in range(n_steps):

            # モデル更新（agent.predict、env.step、agent.updateの一括実行に相当）
            losses, rewards, _ = agent.update_model(sess, state)
            loss, vloss, aloss = losses
            reward, tour_dist = rewards

            list_losses.append([loss, vloss, aloss])
            list_rewards.append(reward)

        # --- POST-PROCESS (EPISODE) ---
        i_episode += 1
        if not i_episode % 50:

            # monitoring metricsの算出
            duration = time.time() - start_time
            avg_loss, avg_vloss, avg_aloss = np.mean(list_losses, axis=0)
            avg_reward = np.mean(list_rewards)

            # monitoringのリセット
            list_losses, list_rewards = [], []
            start_time = time.time()

            # print
            log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode, n_episodes)
            log_str += ' - Time: {0:3.2f}'.format(duration)
            log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
            log_str += ' - Avg_Loss: {0:3.5f}'.format(avg_loss)
            log_str += ' - Avg_VLoss: {0:3.5f}'.format(avg_vloss)
            log_str += ' - Avg_ALoss: {0:3.5f}'.format(avg_aloss)
            print(log_str)

            # modelのlogging
            if not min_metric:
                min_metric = avg_reward
            min_metric = max(min_metric, avg_reward)

            if min_metric is avg_reward:
                args = [i_episode, avg_reward, avg_loss]
                agent.save_graph(sess, log_dir, args)

            # 各種monitoring metricsのlogging
            log_list = [i_episode, avg_reward, avg_loss, avg_vloss, avg_aloss]
            logger.history_save(log_list)


if __name__ == '__main__':
    train()
