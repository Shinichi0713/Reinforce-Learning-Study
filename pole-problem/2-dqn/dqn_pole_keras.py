
import gym
import numpy as np
import time, os
import math
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

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
        
    def train(self, agent):
        num_episodes = 300  # 総試行回数
        max_number_of_steps = 200  # 1試行のstep数
        goal_average_reward = 195  # この報酬を超えると学習終了
        num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
        total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
        gamma = 0.99    # 割引係数
        islearned = 0  # 学習が終わったフラグ
        isrender = 0  # 描画フラグ
        # ---
        
        memory_size = 10000            # バッファーメモリの大きさ
        batch_size = 32                # Q-networkを更新するバッチの大記載
        memory = Memory(max_size=memory_size)
        for episode in range(num_episodes):  # 試行数分繰り返す
            state = self.__init_env()       # 学習のエピソードごとに環境初期化
            episode_reward = 0

            for t in range(max_number_of_steps + 1):  # 1試行のループ
                # if (islearned == 1):  # 学習終了したらcartPoleを描画する
                #     self.env.render()
                #     time.sleep(0.1)
                #     print(state[0, 0])
                action = agent.get_action(state, episode)   # 時刻tでの行動を決定する
                next_state, reward, done, info, _ = self.env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
                next_state = np.reshape(next_state, [1, 4])     # list型のstateを、1行4列の行列に変換
                # 報酬を設定し、与える
                if done:
                    next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
                    if t < 195:
                        reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
                    else:
                        reward = 1  # 立ったまま195step超えて終了時は報酬
                else:
                    reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）
                episode_reward += reward

                memory.add((state, action, reward, next_state))     # メモリの更新する
                state = next_state  # 状態更新

                if (memory.len() > batch_size) and not islearned:
                    agent.replay(memory, batch_size, gamma, agent)
                
                # 1施行終了時の処理
                if done:
                    total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
                    print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                    break
            # 収束判定
            if total_reward_vec.mean() >= goal_average_reward:
                print('Episode %d train agent successfuly!' % episode)
                islearned = 1
                if isrender == 0:   # 学習済みフラグを更新
                    isrender = 1
                agent.save_nn()

    def play(self):
        self.env = gym.make("CartPole-v0", render_mode="human")
        episode = 1e6
        state = self.__init_env()       # 学習のエピソードごとに環境初期化
        for step in range(1000):
            self.env.render()
            action = agent.get_action(state, episode)
            next_state, reward, done, info, _ = self.env.step(action)
            state = np.reshape(next_state, [1, 4])
            self.env.render()
        self.env.close()


    def __init_env(self):
        self.env.reset()  # cartPoleの環境初期化
        observation, reward, done, info, _ = self.env.step(self.env.action_space.sample())  # 1step目は適当な行動をとる
        state = np.reshape(observation, [1, 4])   # list型のstateを、1行4列の行列に変換
        return state



if __name__ == "__main__":
    print("start dqn pole problem")
    is_train = False
    if is_train:
        learning_rate = 0.00001         # Q-networkの学習係数
        agent = Agent(learning_rate)
        env = Env()
        env.train(agent)
    else:
        agent = Agent()
        env = Env()
        env.play()

    ## 参考サイト
    # https://note.com/e_dao/n/n8228e4897bcf
    # https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9