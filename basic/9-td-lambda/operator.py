
import numpy as np
from environment import Environment
from agent import TdLambdaAgent


def train(num_episodes=1000, max_steps=100):
    env = Environment()
    agent = TdLambdaAgent(env)

    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        agent.reset_eligibility_trace()

        for step in range(max_steps):
            action = agent.select_action(state)
            reward, next_state, done = env.step(state, action)
            rewards.append(reward)
            agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards[-100:])}")
            rewards = []


if __name__ == "__main__":
    train(num_episodes=1000, max_steps=100)
    print("Training completed.")