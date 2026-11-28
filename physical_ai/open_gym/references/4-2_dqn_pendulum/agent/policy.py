import numpy as np


class EpsilonGreedyPolicy:

    def __init__(self, q_network, epsilon):
        self.q_network = q_network
        self.epsilon = epsilon

    def get_action(self, state, actions_list):
        is_random_action = (np.random.uniform() <
                            self.epsilon)
        if is_random_action:
            q_values = None
            action = np.random.choice(actions_list)
        else:
            state = np.reshape(state, (1, len(state)))
            q_values = self.q_network.main_network.predict_on_batch(
                state)[0]
            action = actions_list[np.argmax(q_values)]
        return action, self.epsilon, q_values
