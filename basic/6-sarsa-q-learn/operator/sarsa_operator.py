# Sarsaエージェントを操作するコード
import os, sys
dir_current = os.path.dirname(os.path.abspath(__file__))
dir_parent = os.path.dirname(dir_current)
sys.path.append(dir_parent)

import environment
from agent import SarsaAgent

# エージェントを訓練する関数
def train_agent():
    env = environment.Environment()
    agent = SarsaAgent(env)
    for episode in range(1000):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            reward, state_next, done = env.step(state, action)
            next_action = agent.choose_action(state_next)
            agent.update(state, action, reward, state_next, next_action)
            state = state_next
            action = next_action
            if done:
                break
    # 行動価値関数保存
    agent.save()

# エージェントを評価する関数
def evaluate_agent():
    env = environment.Environment()
    agent = SarsaAgent(env)
    for episode in range(100):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            reward, state_next, done = env.step(state, action)
            next_action = agent.choose_action(state_next)
            agent.update(state, action, reward, state_next, next_action)
            state = state_next
            action = next_action
            if done:
                break
    # 行動価値関数保存
    agent.save()

if __name__ == "__main__":
    train_agent()

