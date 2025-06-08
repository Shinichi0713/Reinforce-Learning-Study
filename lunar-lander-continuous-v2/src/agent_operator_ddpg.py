from agent import AgentDdpg
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

def train():
    env = Environment()
    # (self, alpha, beta, tau, env, batch_size)
    agent = AgentDdpg(alpha=0.000025, beta=0.00025, tau=0.001, env=env, batch_size=64)

    np.random.seed(0)
    score_history = []

    for i in range(1000):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            # 行動出力
            act = agent.choose_action(obs)
            # 環境からの応答を取得
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            # 経験メモリー
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))
        if i % 25 == 0:
            agent.save_models()

    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

def load_trained():
    env = Environment(is_train=False)

    agent = AgentDdpg(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=4)
    agent.load_models()

    np.random.seed(0)
    score_history = []

    for i in range(50):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))


    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()




if __name__ == "__main__":
    train()
    # load_trained()