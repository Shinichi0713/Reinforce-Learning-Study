import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque


# 経験再生用のメモリ
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, experience):
        self.buffer.append(experience)
    
    # バッチサイズ分の経験をランダムに取得
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
 
    def len(self):
        return len(self.buffer)


# エージェント
class Agent():
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=100):
        print("model initialized")
        # モデルはkeras製
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(learning_rate=learning_rate)  # 誤差を減らす学習方法はAdam
        # モデルコンパイル
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        print(self.model.summary())
        # ネットワークパラメータパス
        dir_currnet = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = f"{dir_currnet}/nn_parameter.weights.h5"

        self.load_nn()

    # 重みの学習
    def replay(self, memory, batch_size, gamma):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        # バッチサイズ分の経験を取得
        mini_batch = memory.sample(batch_size)
        # 学習サイクル
        # 状態、アクション、報酬、次の状態の取得
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
 
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                ret_model = self.model.predict(next_state_b)[0]
                next_action = np.argmax(ret_model)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * self.model.predict(next_state_b)[0][next_action]
                
            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
 
    # 行動選択
    def get_action(self, state, episode):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)

        # エピソードによりmodelチョイスの比率を調整(通常の方法)
        if epsilon <= np.random.uniform(0, 1):
            logits = self.model.predict(state)[0]
            action = np.argmax(logits)  # 最大の報酬を返す行動を選択する
 
        else:
            action = np.random.choice([0, 1])  # ランダムに行動する
 
        return action

    def save_nn(self):
        print("save nn parameter")
        self.model.save_weights(self.path_nn)
    
    def load_nn(self):
        if os.path.exists(self.path_nn):
            print("load nn parameter")
            self.model.load_weights(self.path_nn)
        else:
            print("start default nn")

class Env():
    def __init__(self):
        self.env = gym.make("CartPole-v0")

    def __init_env(self):
        self.env.reset()  # cartPoleの環境初期化
        observation, reward, done, info, _ = self.env.step(self.env.action_space.sample())  # 1step目は適当な行動をとる
        state = np.reshape(observation, [1, 4])   # list型のstateを、1行4列の行列に変換
        return state


