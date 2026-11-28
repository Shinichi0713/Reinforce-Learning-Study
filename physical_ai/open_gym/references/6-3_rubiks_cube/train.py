"""
overview:
    Actor-Critic法によるルービックキューブの解検索

args:
    各種パラメータ規定値は、本コード中に明記される
        - log_dir: 下記要素の出力先
        - model_path: 初期設定用学習済みモデルファイル（任意）
        - n_episodes: 学習エピソード数
        - n_steps: 操作ステップ数

output:
    上記log_dirに以下の要素を出力する
        - checkpoint: チェックポイントファイル
        - model.xxx-xxx.ckpt: チェックポイント毎のモデルファイル
        - reward_log.csv: 報酬、損失など監視指標のログ

usage-example:
    python3 train.py --log_dir=./result \
    --n_episodes=150000 --n_steps=15
"""
import os
import argparse
import time
import csv

import numpy as np
import tensorflow as tf

from agent.actor_critic_agent import ActorCriticAgent
from agent.memory import Memory
from gym_env.rubiks_cube_env import RubiksCubeEnv


class HistoryLogger(object):

    def __init__(self, log_dir):
        self.history_path = os.path.join(
            log_dir, 'reward_log.csv')

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
    parser.add_argument('--log_dir',
                        type=str,
                        default='./result',
                        help='log directory')
    parser.add_argument('--model_path',
                        type=str,
                        default='',
                        help='path to model ckpt file')
    parser.add_argument('--n_episodes',
                        type=int,
                        default=150000,
                        help='# of episodes for train')
    parser.add_argument('--n_steps',
                        type=int,
                        default=15,
                        help='# of steps for train')

    return parser.parse_args()


def train():
    # argsの取得
    args = get_args()
    log_dir = args.log_dir
    model_path = args.model_path
    n_episodes = args.n_episodes
    n_steps = args.n_steps

    os.makedirs(args.log_dir, exist_ok=True)

    # メインの処理プロセス
    # --- PRE-PROCESS ---
    # セッションスタート
    sess = tf.Session()

    # インスタンスの作成
    env = RubiksCubeEnv()
    st_shape, act_list =\
        env.get_state_shape(), env.get_action_list()
    agent = ActorCriticAgent(st_shape, act_list)
    memory = Memory()
    logger = HistoryLogger(log_dir)

    # ネットワーク変数の初期化
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # 学習済みモデルからネットワーク変数をrestore
    if model_path:
        agent.restore_graph(sess, model_path)

    # history loggingのヘッダ定義
    _header = [
        'episode', 'avg_reward', 'avg_loss', 'avg_vloss',
        'avg_aloss'
    ]
    logger.set_history_header(_header)

    # --- TRAIN MAIN ---
    # monotoring metrics用の変数
    min_metric = 0.0
    list_losses, list_rewards = [], []
    start_time = time.time()

    # エピソードのループ
    for i_episode in range(n_episodes):

        # Cube環境の初期化
        env.reset()
        # Cubeのランダムシャッフル
        _, state = env.apply_scramble_w_weight()

        # ステップのループ
        for i_step in range(n_steps):

            # エージェント(方策ネットワーク)による行動推定
            action = agent.get_action(sess, state)

            # 選択行動に対して、環境から報酬値などの取得
            next_state, reward, done, _ = env.step(action)

            # 経験のメモリ登録
            memory.push(state, action, reward,
                        next_state, done)

            state = next_state

            if done[0]:
                break

        # --- POST-PROCESS (EPISODE) ---
        # メモリからの経験データの取得
        memory_data = memory.get_memory_data()

        # 経験データを用いたエージェントの更新
        _args = zip(*memory_data)
        losses = agent.update_model(sess, *_args)

        loss, vloss, aloss = losses
        _, _, _rwd, _, _ = zip(*memory_data)
        reward = _rwd

        list_losses.append([loss, vloss, aloss])
        list_rewards.append(np.mean(reward))

        # 次エピソードのためメモリの初期化
        memory.reset()

        i_episode += 1
        if not i_episode % 100:

            # monitoring metricsの算出
            duration = time.time() - start_time
            avg_loss, avg_vloss, avg_aloss = np.mean(
                list_losses, axis=0)
            avg_reward = np.mean(list_rewards)

            # monitoringのリセット
            list_losses, list_rewards = [], []
            start_time = time.time()

            # print
            log_str = 'Episode: {0:6d}/{1:6d}'.format(
                i_episode, n_episodes)
            log_str += ' - Time: {0:3.2f}'.format(
                duration)
            log_str += ' - Avg_Reward: {0:3.3f}'.format(
                avg_reward)
            log_str += ' - Avg_Loss: {0:3.5f}'.format(
                avg_loss)
            log_str += ' - Avg_VLoss: {0:3.5f}'.format(
                avg_vloss)
            log_str += ' - Avg_ALoss: {0:3.5f}'.format(
                avg_aloss)
            print(log_str)

            # modelのlogging
            if not min_metric:
                min_metric = avg_reward
            min_metric = max(min_metric, avg_reward)

            if min_metric is avg_reward:
                args = [i_episode, avg_reward, avg_loss]
                agent.save_graph(sess, log_dir, args)

            # 各種monitoring metricsのlogging
            log_list = [
                i_episode, avg_reward, avg_loss,
                avg_vloss, avg_aloss
            ]
            logger.history_save(log_list)


if __name__ == '__main__':
    train()
