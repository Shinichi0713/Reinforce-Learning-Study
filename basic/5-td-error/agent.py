


class Agent:
    def __init__(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor


    def select_action(self, state, action):

        pass


if __name__ == "__main__":
    agent_instance = Agent(learning_rate=0.01, discount_factor=0.99)

