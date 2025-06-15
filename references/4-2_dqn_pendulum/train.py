"""
overview:
    OpenAI GymのPendulum-v0を環境として、Double_DQNの学習を行う

args:
    各種パラメータの設定値は、本コード中に明記される
    - result_dir:
        結果を出力するディレクトリのpath
    - max_episode:
        学習の繰り返しエピソード数(default: 300)
    - max_step:
        1エピソード内の最大ステップ数(default: 200)
    - gamma:
        割引率(default: 0.99)
output:
    result_dirで指定したpathに以下のファイルが出力される
    - episode_xxx.h5:
        xxxエピソードまで学習したDouble_DQNネットワークの重み
    - history.csv: エピソードごとの以下の3つのメトリックを記録するcsv
        - loss: DoubleDQNモデルを更新する際のlossの平均値
        - td_error: TD誤差の平均値
        - reward_avg: １ステップあたりの平均報酬

usage：
    python3 train.py
"""
import os
import random

import gym
import numpy as np

from agent.model import Qnetwork
from agent.policy import EpsilonGreedyPolicy
from util import now_str, RecordHistory


def train():
    # setup ===========================
    max_episode = 300  # 学習において繰り返す最大エピソード数
    max_step = 200  # 1エピソードの最大ステップ数
    n_warmup_steps = 10000  # warmupを行うステップ数
    interval = 1  # モデルや結果を吐き出すステップ間隔
    actions_list = [-1, 1]  # 行動(action)の取りうる値のリスト
    gamma = 0.99  # 割引率
    epsilon = 0.1  # ε-greedyのパラメータ
    memory_size = 10000
    batch_size = 32
    result_dir = os.path.join('./result/pendulum',
                              now_str())

    # インスタンス作成 ==================
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)
    env = gym.make('Pendulum-v0')
    dim_state = env.env.observation_space.shape[0]
    q_network = Qnetwork(dim_state,
                         actions_list,
                         gamma=gamma)
    policy = EpsilonGreedyPolicy(q_network,
                                 epsilon=epsilon)
    header = [
        "num_episode", "loss", "td_error", "reward_avg"
    ]
    recorder = RecordHistory(
        os.path.join(result_dir, "history.csv"), header)
    recorder.generate_csv()

    # warmup=======================
    print('warming up {:,} steps...'.format(
        n_warmup_steps))
    memory = []
    total_step = 0
    step = 0
    state = env.reset()
    while True:
        step += 1
        total_step += 1

        action = random.choice(actions_list)
        epsilon, q_values = 1.0, None

        next_state, reward, done, info = env.step(
            [action])

        # reward clipping
        if reward < -1:
            c_reward = -1
        else:
            c_reward = 1
        memory.append(
            (state, action, c_reward, next_state, done))
        state = next_state

        if step > max_step:
            state = env.reset()
            step = 0
        if total_step > n_warmup_steps:
            break
    memory = memory[-memory_size:]
    print('warming up {:,} steps... done.'.format(
        n_warmup_steps))

    # training======================
    print(
        'training {:,} episodes...'.format(max_episode))
    num_episode = 0
    episode_loop = True
    while episode_loop:
        num_episode += 1
        step = 0
        step_loop = True
        episode_reward_list, loss_list, td_list = [], [], []
        state = env.reset()

        while step_loop:
            step += 1
            total_step += 1
            action, epsilon, q_values = policy.get_action(
                state, actions_list)
            next_state, reward, done, info = env.step(
                [action])

            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1

            memory.append((state, action, c_reward,
                           next_state, done))
            episode_reward_list.append(c_reward)
            exps = random.sample(memory, batch_size)
            loss, td_error = q_network.update_on_batch(
                exps)
            loss_list.append(loss)
            td_list.append(td_error)

            q_network.sync_target_network(soft=0.01)
            state = next_state
            memory = memory[-memory_size:]

            # end of episode
            if step >= max_step:
                step_loop = False
                reward_avg = np.mean(episode_reward_list)
                loss_avg = np.mean(loss_list)
                td_error_avg = np.mean(td_list)
                print(
                    "{}episode  reward_avg:{} loss:{} td_error:{}"
                    .format(num_episode, reward_avg,
                            loss_avg, td_error_avg))
                if num_episode % interval == 0:
                    model_path = os.path.join(
                        result_dir,
                        'episode_{}.h5'.format(
                            num_episode))
                    q_network.main_network.save(
                        model_path)
                    history = {
                        "num_episode": num_episode,
                        "loss": loss_avg,
                        "td_error": td_error_avg,
                        "reward_avg": reward_avg
                    }
                    recorder.add_histry(history)

        if num_episode >= max_episode:
            episode_loop = False

    env.close()
    print('training {:,} episodes... done.'.format(
        max_episode))


if __name__ == '__main__':
    train()
