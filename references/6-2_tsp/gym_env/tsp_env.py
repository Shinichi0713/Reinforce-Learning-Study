import numpy as np

import gym


class TSPEnv(gym.Env):

    def __init__(self, train_flg, batch_size=4, seq_length=10, coord_dim=2):
        self.train_flg = train_flg

        # setup consts
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.coord_dim = coord_dim

        # data format
        self.data_coord_max = 100.

    def seed(self, seed=None):
        pass

    def render(self, mode='human', close=False):
        pass

    def get_data(self):
        return self.data

    # NOTE: receive route/tour prediction (array) as action
    def step(self, action):

        tour = action
        reward = -1.0 * self._get_distance(self.state, tour)

        # sort data by the tour
        self.state = self._batch_seq_sort(self.state, tour[:, :-1])
        self.data = self._batch_seq_sort(self.data, tour[:, :-1])

        return self.state, reward, True, {}

    # TODO: too redundant, could be simplified w/ proper numpy functions
    def _batch_seq_sort(self, btch_seq, btch_order):

        btch_size = btch_seq.shape[0]
        btch_order_seq = [btch_seq[i, btch_order[i]] for i in range(btch_size)]

        return np.array(btch_order_seq)

    def _get_distance(self, data, tour):
        # order input_data by tour
        _sort_data = self._batch_seq_sort(data, tour)
        # [n_batch, n_seq', n_dim] to [n_dim, n_seq', n_batch]
        # n_seq' = n_seq+1; because end (=start) location explicitly included
        _sort_data = np.transpose(_sort_data, (2, 1, 0))

        # ordered coordinates
        sort_x = _sort_data[0]
        sort_y = _sort_data[1]  # [n_seq', n_batch]

        # euclidean distance btw locations
        dlt_x2 = np.square(sort_x[1:] - sort_x[:-1])
        dlt_y2 = np.square(sort_y[1:] - sort_y[:-1])
        inter_distances = np.sqrt(dlt_x2 + dlt_y2)  # [n_seq', batch_size]

        distance = np.sum(inter_distances, axis=0)  # [batch_size]

        return distance

    # NOTE: different format depending on train_flg
    def reset(self):

        if self.train_flg:

            batch_norm_data, batch_data = [], []
            for _ in range(self.batch_size):

                norm_data, data = self._generate_data()
                batch_norm_data.append(norm_data)
                batch_data.append(data)

        else:
            norm_data, data = self._generate_data()

            batch_norm_data = np.tile(norm_data, (self.batch_size, 1, 1))
            batch_data = np.tile(data, (self.batch_size, 1, 1))

        self.state = np.array(batch_norm_data)
        self.data = np.array(batch_data)

        return self.state

    # generate location data
    def _generate_data(self):
        # data shape
        data_shape = (self.seq_length, self.coord_dim)

        # randomly generate city locations with dimension
        data = np.random.randint(self.data_coord_max, size=data_shape)

        norm_data = self._format_data(data)

        return norm_data, data

    def _format_data(self, data):
        # normalize coordinates
        norm_data = data - np.mean(data, axis=0)
        norm_data /= self.data_coord_max

        return norm_data


def main():

    env = TSPEnv(train_flg=True)

    state = env.reset()
    print(state)

    # make temp tour
    _size = state.shape[0]
    _lgth = state.shape[1]
    tour = [np.random.permutation(np.arange(_lgth)) for i in range(_size)]
    tour = [np.hstack((tour[i], tour[i][0])) for i in range(_size)]
    tour = np.array(tour)
    print(tour)

    next_state, reward, _, _ = env.step(tour)
    print(next_state, reward)


if __name__ == "__main__":
    main()
