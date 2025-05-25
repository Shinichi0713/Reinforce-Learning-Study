
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
            s_idx = agent.state_to_idx(state, env.maze.shape[1])
            action = agent.select_action(s_idx)
            reward, next_state, done = env.step(state, agent.actions[action])
            rewards.append(reward)
            # s_idx, a, r, s_next_idx, a_next, done
            s_idx_next = agent.state_to_idx(next_state, env.maze.shape[1])
            agent.update(s_idx, action, reward, s_idx_next, done)

            state = next_state

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards[-100:])}")
            rewards = []


if __name__ == "__main__":
    train(num_episodes=1000, max_steps=100)
    print("Training completed.")