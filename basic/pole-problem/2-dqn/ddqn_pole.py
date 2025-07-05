
import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
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
        self.buffer = deque(maxlen=size_max)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def len(self):
        return len(self.buffer)

# エージェント
class Agent():
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=100):
        # モデルはkeras製
        self.model=nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.is_prepare_train = False
        
    def train_replay(self, experience, size_batch, gamma):
        # train準備
        self.prepare_train()

        if len(experience) < size_batch:
            return
        mini_batch =experience.sample(size_batch)
        # バッチ分の学習
        for i, (state, action, reward, next_state) in enumerate(mini_batch):
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            # if not (next_state_b == np.zeros(state_b.shape)).all(axis=1): は、次の状態 next_state_b がゼロの状態であるかどうかをチェック
            if not (next_state == torch.zeros(state.shape).all(axis=1)):
                ret_model = self.model.predict(next_state)[0]
                next_action = np.argmax(ret_model)  # Q学習的な解釈
                # Q学習の更新式適用
                target = reward + gamma * torch.max(self.model(next_state)[0][next_action]).item()
            predict = self.model(state)
            target_f = predict.clone()
            # バッチ化されても、データ数が1→予測を次の価値関数に更新
            target_f[0][action] = target
            self.optimizer.zero_grad()
            # ロスは状態から予測された行動価値と、次の状態から予測された行動価値の差
            loss = criterion(target_f, predict)
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def prepare_train(self):
        if not self.is_prepare_train:
            self.epsilon = 0.1
            self.epsilon_min = 1e-6

            self.is_prepare_train = True

if __name__ == "__main__":
    x_true = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    x_pred = torch.tensor([0.5, 2.5, 2.5, 4], dtype=torch.float32)

    criterion = HuberLoss()
    loss = criterion(x_true, x_pred)
    print(loss)
