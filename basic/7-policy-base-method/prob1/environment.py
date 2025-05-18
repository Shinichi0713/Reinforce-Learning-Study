## CartPoleを用いた環境
import gym

class PoleGym:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode="human")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = PoleGym()
    state = env.reset()
    done = False
    for i in range(20):
        
        action = env.env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        state = next_state
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(i+1))
            break
    env.close()
