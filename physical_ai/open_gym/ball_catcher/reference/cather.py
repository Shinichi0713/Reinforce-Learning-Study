import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from ple import PLE
from ple.games.catcher import Catcher

# 環境の作成
game = Catcher(width=256, height=256)
env = PLE(game, fps=30, display_screen=True)
env.init()

# パラメータ
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
episodes = 300
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

# Qネットワークの構築
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

model = build_model()

# 経験リプレイで学習
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state[np.newaxis, :], verbose=0)[0])
        target_f = model.predict(state[np.newaxis, :], verbose=0)
        target_f[0][action] = target
        model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

# メインループ
for e in range(episodes):
    state = env.reset()
    state = np.array(state)
    total_reward = 0
    done = False
    while not done:
        # env.render()  # 描画したい場合はコメントを外す
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])
        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        replay()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

env.close()
