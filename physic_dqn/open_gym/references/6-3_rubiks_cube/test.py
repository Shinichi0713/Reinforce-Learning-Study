"""
overview:
    学習済みモデルとMCTSを用いたルービックキューブの解予測

args:
    各種パラメータ規定値は、本コード中に明記される
        - log_dir: 下記要素の出力先
        - model_path: 予測用学習済みモデルファイル
        - n_episodes: 予測エピソード数
        - n_steps: 操作ステップ数

output:
    上記log_dirに以下の要素を出力する
        - list_results.pkl: 予測結果のリスト

usage-example:
    python3 train.py --log_dir=./result \
    --model_path=./results/model.099800-0.394-2.65370.ckpt \
    --n_episodes=5000 --n_steps=15
"""
import os
import argparse
import time
import pickle

import numpy as np
import tensorflow as tf

from agent.actor_critic_agent import ActorCriticAgent
from gym_env.rubiks_cube_env import RubiksCubeEnv
from util.mcts import MCTS


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
    parser.add_argument('--n_steps', type=int, default=15,
                        help='# of steps for test')

    return parser.parse_args()


def test():
    # argsの取得
    args = get_args()
    log_dir = args.log_dir
    model_path = args.model_path
    n_episodes = args.n_episodes
    n_steps = args.n_steps

    os.makedirs(args.log_dir, exist_ok=True)

    # --- PRE-PROCESS ---
    # tf sessionのスタート
    sess = tf.Session()

    # インスタンスの作成
    env = RubiksCubeEnv()
    st_shape, act_list = env.get_state_shape(), env.get_action_list()
    agent = ActorCriticAgent(st_shape, act_list)
    mcts = MCTS(agent)
    logger = ObjectLogger(log_dir)

    # ネットワーク変数の初期化
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # 学習済みモデルからネットワーク変数をrestore
    agent.restore_graph(sess, model_path)

    # --- TEST MAIN ---
    # monotoring metrics用の変数
    list_losses, list_rewards = [], []
    list_summary = []
    start_time = time.time()

    # episideのループ
    for i_episode in range(n_episodes):

        # Cube環境の初期化
        env.reset()
        # Cubeのランダムシャッフル
        n_scrm, state = env.apply_scramble_wo_weight()
        scramble_hist = env.get_scramble_log()

        n_act, is_slv = n_steps, False
        actions, states = [], [state]

        # MCTS post-processを行う場合
        if 1:
            # mcts searchの実行
            reward, is_solved, states, actions = mcts.run_search(sess, state)

            list_rewards.append(reward)
            n_act, is_slv = len(actions), is_solved

            # post-process処理
            i_episode += 1

            values = agent.predict_value(sess, states)
            list_summary.append([n_scrm, scramble_hist, is_slv,
                                 states, values, actions, n_act])

            if not i_episode % 10:
                # monitoring metricsの算出
                duration = time.time() - start_time
                avg_reward = np.mean(list_rewards)

                # monitoringのリセット
                list_rewards = []
                start_time = time.time()

                # print
                log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode,
                                                          n_episodes)
                log_str += ' - Time: {0:3.2f}'.format(duration)
                log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
                print(log_str)

        # MCTS post-processを行わない場合
        if 0:
            # stepのループ
            for i_step in range(n_steps):

                # エージェント(方策ネットワーク)による行動推定
                action = agent.get_action(sess, state)

                # 選択行動に対して、環境から報酬値などの取得
                next_state, reward, done, _ = env.step(action)

                # loss推定値の取得
                args = ([state], [action], [reward], [next_state], [done])
                losses = agent.predict_loss(sess, *args)
                loss, vloss, aloss = losses

                list_losses.append([loss, vloss, aloss])
                list_rewards.append(reward)

                state = next_state

                actions.append(action)
                states.append(state)

                if done[0]:
                    n_act, is_slv = i_step + 1, True
                    break

            # --- POST-PROCESS (EPISODE) ---
            i_episode += 1

            values = agent.predict_value(sess, states)
            list_summary.append([n_scrm, scramble_hist, is_slv,
                                 states, values, actions, n_act])

            if not i_episode % 100:

                # monitoring metricsの算出
                duration = time.time() - start_time
                avg_loss, avg_vloss, avg_aloss = np.mean(list_losses, axis=0)
                avg_reward = np.mean(list_rewards)

                # monitoringのリセット
                list_losses, list_rewards = [], []
                start_time = time.time()

                # print
                log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode,
                                                          n_episodes)
                log_str += ' - Time: {0:3.2f}'.format(duration)
                log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
                log_str += ' - Avg_Loss: {0:3.5f}'.format(avg_loss)
                log_str += ' - Avg_VLoss: {0:3.5f}'.format(avg_vloss)
                log_str += ' - Avg_ALoss: {0:3.5f}'.format(avg_aloss)
                print(log_str)

    # --- POST-PROCESS (TESTING) ---
    # 予測結果リストのオブジェクトの出力
    log_obj = zip(*list_summary)
    logger.object_save(log_obj)


if __name__ == '__main__':
    test()
