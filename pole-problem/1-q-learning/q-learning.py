# https://github.com/Fumio-eisan/RL20200527/blob/master/CartPole_20200526.ipynb
import gym
import numpy as np
import time, os
import math
from statistics import mean, median
import matplotlib.pyplot as plt


# pole問題のエージェント
# オフポリシー学習ベースの学習→学習した結果をplayで再現
class Agent:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        # 観察値の等差数列(離散化する)
        self.bins = {
            'cart_position': np.linspace(-2.4, 2.4, 3)[1:-1],
            'cart_velocity': np.linspace(-3, 3, 5)[1:-1],
            'pole_angle':    np.linspace(-0.5, 0.5, 5)[1:-1],
            'pole_velocity': np.linspace(-2, 2, 5)[1:-1]
        }
        # 状態の設定
        num_state = (
            (len(self.bins['cart_position']) + 1)
            * (len(self.bins['cart_velocity']) + 1)
            * (len(self.bins['pole_angle']) + 1)
            * (len(self.bins['pole_velocity']) + 1)
        )
        self.q_table = np.random.uniform(low=-1, high=1, size=(num_state, self.env.action_space.n))
        # qテーブルの更新履歴(一応取っておく。これ自体はQ学習云々とはかかわりなし)
        self.q_table_update = [[math.floor(time.time()), 0]] * num_state

        # Qテーブルの保存先。基準はカレント
        self.dir_current = os.path.dirname(os.path.abspath(__file__))
        self.__load_q_table()

    # 状態値の取得
    def __digitize_state(self, observation):
        """
        状態のデジタル化をする
        :param observation: 観察値
        :return: デジタル化した数値
        """
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        state = (
            np.digitize(cart_position, bins=self.bins['cart_position'])
            + (
                np.digitize(cart_velocity, bins=self.bins['cart_velocity'])
                * (len(self.bins['cart_position']) + 1)
            )
            + (
                np.digitize(pole_angle, bins=self.bins['pole_angle'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
            )
            + (
                np.digitize(pole_velocity, bins=self.bins['pole_velocity'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
                * (len(self.bins['pole_angle']) + 1)
            )
        )
        return state

    def get_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        # ε-greedy法で行動を選択
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice([0, 1])
        return action

    def update_q_table(self, state, action, reward, state_next):
        alpha = 0.2
        gamma = 0.99
        q_value_max = max(self.q_table[state_next])
        q_value_current = self.q_table[state, action]
        # qテーブルの更新＝Q学習で更新
        self.q_table[state, action] = q_value_current \
            + alpha * (reward + gamma * q_value_max - q_value_current)
        self.q_table_update[state] = [math.floor(time.time()), self.q_table_update[state][1] + 1]

    def train(self, num_episode, num_step_max):
        num_solved = 0
        num_solved_max = 0

        # 履歴用
        self.episodes = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.episode_rewards_median = []

        for episode in range(num_episode):
            print(f"Episode: {episode}")
            observation, _ = self.env.reset()
            state = self.__digitize_state(observation)
            action = np.argmax(self.q_table[state])
            episode_reward = 0

            for step in range(num_step_max):
                observation, reward, done, info, _ = self.env.step(action)
            
                if done and step < num_step_max - 1:
                    reward -= num_step_max
                episode_reward += reward

                state_next = self.__digitize_state(observation)
                self.update_q_table(state, action, reward, state_next) 
                # 行動の選択
                action = self.get_action(state_next, episode)
                state = state_next
                
                # 学習の終了
                if done:
                    self.__save_q_table()
                    break

    def __save_q_table(self):
        print("Save q_table")
        np.save(f"{self.dir_current}/q_table.npy", self.q_table)

    def __load_q_table(self):
        if os.path.exists(f"{self.dir_current}/q_table.npy"):
            print("Load q_table")
            self.q_table = np.load(f"{self.dir_current}/q_table.npy")

    # 学習したエージェントでプレイ
    def play(self):
        # レンダリングするためモードチェンジ
        self.env = gym.make("CartPole-v0", render_mode="human")
        observation, _ = self.env.reset()
        state = self.__digitize_state(observation)
        action = np.argmax(self.q_table[state])

        for step in range(1000):
            self.env.render()
            state = self.__digitize_state(observation)
            action = np.argmax(self.q_table[state])
            observation, reward, done, info, _ = self.env.step(action)
            self.env.render()
        self.env.close()


if __name__ == "__main__":
    agent = Agent()
    # agent.train(100, 200)
    agent.play()
