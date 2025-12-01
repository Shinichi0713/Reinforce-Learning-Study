import gym
import numpy as np
import copy

from tensorflow.keras.utils import to_categorical

from gym_env.cube_algorithm import Cube, Cube3x3, Cube2x2

env_config = {
    'order': 2,
    'action_list': ['f', '.f', 'r', '.r', 'u', '.u'],
    'inv_action_list': ['.f', 'f', '.r', 'r', '.u', 'u'],
    'reward_solved': +1.0,
    'reward_unsolved': -1.0,
    'min_actions': 1,
    'max_actions': 10,
    'scramble_power': 1.0
}


class RubiksCubeEnv(gym.Env):

    # constructor
    def __init__(self):
        # cube order
        order = env_config['order']

        self.cube = Cube3x3() if order == 3 else Cube2x2()
        self.action_list = env_config['action_list']
        self.inv_action_list = env_config['inv_action_list']

        color_dict = Cube.color_id
        self.num_color = len(color_dict.keys())

        self.state = self.get_state()

        # reward config
        self.rwd_solved = env_config['reward_solved']
        self.rwd_unsolved = env_config['reward_unsolved']

        # setup scramble variables
        self.act_min = env_config['min_actions']
        self.act_max = env_config['max_actions']
        self.scrm_list = range(self.act_min, self.act_max + 1)
        # inverse weight
        power = env_config['scramble_power']
        self.scrm_weight = [1.0/(scrm**power) for scrm in self.scrm_list]
        self.scrm_weight /= np.sum(self.scrm_weight)

        # history logs
        self.action_log = []
        self.scramble_log = []

    # take action step and update cube state
    def step(self, action):

        self.cube.apply_action(action)
        self.action_log.append(action)

        self.state = self.get_state()

        if self.cube.is_solved():
            reward = self.rwd_solved
            done = True
        else:
            reward = self.rwd_unsolved
            done = False

        # keep dim. full for reward & done
        return self.state, [reward], [done], {}

    # initialize cube state w/ scrambles
    def reset(self):
        self.cube.reset_cube()

        self.state = self.get_state()

        self.step_count = 0
        self.action_log = []

        return self.state

    # state rendering to stdout
    def render(self, mode='human', close=False):
        if not close:
            self.cube.display_cube()

    # scramble cube and record its history
    def _apply_scramble(self, n_scramble):
        self.scramble_log = []

        for i in range(n_scramble):
            action = np.random.choice(self.action_list)
            self.cube.apply_action(action)
            self.scramble_log.append(action)

        self.state = self.get_state()
        # repeat if cube become solved
        if self.cube.is_solved():
            return self._apply_scramble(n_scramble)

        return self.state

    # scramble cube w/ weight
    def apply_scramble_w_weight(self):
        n_scrm = np.random.choice(self.scrm_list, p=self.scrm_weight)
        state = self._apply_scramble(n_scrm)
        return n_scrm, state

    # scramble cube w/ weight
    def apply_scramble_wo_weight(self):
        n_scrm = np.random.choice(self.scrm_list)
        state = self._apply_scramble(n_scrm)
        return n_scrm, state

    # get one-hot/categorical cube state
    def get_state(self):
        state_vector = self.cube.get_state()
        return copy.deepcopy(to_categorical(state_vector, self.num_color))

    # set cube state from one-hot/categorical cube state
    def set_state(self, cat_state):
        state_vector = np.argmax(cat_state, axis=-1)
        self.cube.set_state(copy.deepcopy(state_vector))

    def is_solved(self):
        return self.cube.is_solved()

    def get_action_list(self):
        return self.action_list

    def get_inv_action_list(self):
        return self.inv_action_list

    def get_state_shape(self):
        return self.state.shape

    def get_scramble_log(self):
        return self.scramble_log


def main():
    env = RubiksCubeEnv()
    env.reset()
    env.render()
    print(env.is_solved())

    env.step('f')
    env.render()

    env.step('r')
    env.render()

    env.step('.u')
    env.render()

    env.step('.b')
    env.render()
    print(env.is_solved())


if __name__ == "__main__":
    main()
