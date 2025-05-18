
import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque


# フーバーロス
class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, y_true, y_pred):
        err = y_true - y_pred
        cond = torch.abs(err) < 1.0
        L2 = 0.5 * err ** 2
        L1 = (torch.abs(err) - 0.5)
        loss = torch.where(cond, L2, L1)
        return torch.mean(loss)

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

# ニューラルネットワークの定義
class Agent():
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=100):
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            # nn.Softmax(dim=1)
        )
        # self.loss_fn = HuberLoss()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = f"{dir_current}/nn_parameter.pth"
        self.__load_nn()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def __load_nn(self):
        if os.path.exists(self.path_nn):
            self.model.load_state_dict(torch.load(self.path_nn))
            print("load model")

    def save_nn(self):
        self.model.eval()
        self.model.cpu()
        torch.save(self.model.state_dict(), self.path_nn)
        self.model.to(self.device)

    def replay_train(self, memory, batch_size, gamma):
        if memory.len() < batch_size:
            return
        self.model.train()
        batch = memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            # テンソルに変換
            state = torch.FloatTensor(state).to(self.device)
            action = torch.LongTensor([action]).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            # あるべき価値関数の計算
            if not done:
                next_q_values = self.model(next_state)
                target = reward + gamma * torch.max(next_q_values).item()
            else:
                target = reward
            target_f = self.model(state)
            target_f[0][action] = target
            loss = self.loss_fn(target_f, self.model(state))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_action(self, state, epsilon):
        if np.random.rand() > epsilon:
            return np.random.choice(2)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        

# 環境
class Env():
    def __init__(self):
        self.env = gym.make('CartPole-v1')

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
                # reward clip
                # if done:
                #     # next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
                #     if reward < -1:
                #         reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
                #     else:
                #         reward = 1  # 立ったまま195step超えて終了時は報酬
                # else:
                #     reward = 0 
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
                
                # if done:
                #     break
        self.env.close()

if __name__ == "__main__":
    print("start dqn pole problem")
    is_train = False
    if is_train:
        learning_rate = 1e-6         # Q-networkの学習係数
        agent = Agent(learning_rate)
        env = Env()
        env.train(agent)
    else:
        agent = Agent()
        env = Env()
        env.play(agent)