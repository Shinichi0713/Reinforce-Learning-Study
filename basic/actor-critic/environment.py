

import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque


from pathlib import Path


def make_directory():
    dir_current = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = Path(dir_current + '/mdoel')
    result_dir_path = Path(dir_current + '/result')
    if not model_dir_path.exists():
        model_dir_path.mkdir()
    if not result_dir_path.exists():
        result_dir_path.mkdir()

# 経験再生用のメモリ
class Memory:
    def __init__(self, size_max=1000):
        # バッファーの初期化
        self.buffer = deque(maxlen=size_max)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def len(self):
        return len(self.buffer)

# 環境
class Env():
    def __init__(self):
        self.env = gym.make("CartPole-v0")

        # action_size = self.env.action_space.n
        # print("action_size", action_size)

    def train(self, agent, episodes=1000, batch_size=32, gamma=0.99):
        num_episodes = 300  # 総試行回数
        max_number_of_steps = 200  # 1試行のstep数
        goal_average_reward = 195  # この報酬を超えると学習終了
        num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
        total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
        gamma = 0.95    # 割引係数
        islearned = False  # 学習が終わったフラグ
        epsilon = 0.99
        memory_size = 10000            # バッファーメモリの大きさ
        batch_size = 32                # Q-networkを更新するバッチの大記載
        memory = Memory(memory_size)

        for episode in range(num_episodes):  # 試行数分繰り返す
            episode_reward = 0
            state = self.__init_env()
            for t in range(max_number_of_steps + 1):
                action = agent.get_action(state, epsilon)
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                if reward < -1:
                    reward = -1
                elif reward > 1:
                    reward = 1
                else:
                    reward = 0

                episode_reward += reward
                memory.add((state, action, reward, next_state, done)) 
                state = next_state

                if memory.len() > batch_size and not islearned:
                    agent.replay_train(memory, batch_size, gamma)
                    epsilon *= 0.95
            
                # 1施行終了時の処理
                if done:
                    total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
                    print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                    break

            # 収束判断
            # if total_reward_vec.mean() >= goal_average_reward:
            print('Episode %d train agent successfuly!' % episode)
            # islearned = True
            # モデルパラメータ保存
            agent.save_nn()


    def __init_env(self):
        self.env.reset()  # cartPoleの環境初期化
        observation, reward, done, info, _ = self.env.step(self.env.action_space.sample())  # 1step目は適当な行動をとる
        state = np.reshape(observation, [1, 4])   # list型のstateを、1行4列の行列に変換
        return state
    
    def play(self, agent):
        self.env = gym.make("CartPole-v0", render_mode="human")
        agent.model.eval()
        state = self.__init_env()
        with torch.no_grad():
            for _ in range(200):
                self.env.render()
                action = agent.get_action(state, 1.0)
                next_state, reward, done, info, _ = self.env.step(action)
                state = np.reshape(next_state, [1, 4])
        self.env.close()

make_directory()

if __name__ == "__main__":

    env = Env()
    env.env.render()