"""
class implementation of Rubiks-cube manipulations (w/o global rotation)
ref: https://github.com/RobinChiu/gym-Rubiks-Cube
"""
import sys
import numpy as np


class Cube:
    # color id to value
    color_id = {'W': 0, 'G': 1, 'B': 2, 'R': 3, 'O': 4, 'Y': 5}
    # face id to value
    face_id = {'f': 0, 'r': 1, 'l': 2, 'u': 3, 'd': 4, 'b': 5}
    # face id to color id for initial state
    face_to_col = {'f': 'W', 'r': 'O', 'l': 'R', 'u': 'G', 'd': 'B', 'b': 'Y'}

    # edge tile idxs in a face (idx ascend order)
    # (to be overridden)
    edge_tile = {'u_': [], 'd_': [], 'l_': [], 'r_': [],
                 '_u': [], '_d': [], '_l': [], '_r': []}

    # constructor
    def __init__(self, order):
        self.order = order

        # set initial state
        self.cube_state = self._get_initial_state(order)
        # for reference
        self.init_cube_state = self.cube_state.copy()

        # map from rotate op. to associated edge tiles
        self.tile_map = self._get_tile_map()
        # mem. cache of tile array for rotate op.
        self.tile_array = np.zeros((order + 2, order + 2))
        # idx range of edge part in tile array
        self.idx_range = {
            'u': (range(0, 1), range(1, order + 1)),
            'r': (range(1, order + 1), range(order + 1, order + 2)),
            'd': (range(order + 1, order + 2), range(1, order + 1)),
            'l': (range(1, order + 1), range(0, 1))
        }

    # create initial state
    def _get_initial_state(self, order):
        init_state = []

        for face_id in Cube.face_id.keys():
            color_val = Cube.color_id[Cube.face_to_col[face_id]]
            init_state.append(np.full((order, order), color_val))

        return np.array(init_state)

    # create tile map
    def _get_tile_map(self):
        # refer to parent class variable
        _dict = type(self).edge_tile
        # tile idxs of associated edges around a target facek
        # (keep indx ascend order for mapping)
        tile_map = {
            'f': {'u': ('u', _dict['d_']), 'r': ('r', _dict['l_']),
                  'd': ('d', _dict['u_']), 'l': ('l', _dict['r_'])},
            'r': {'u': ('u', _dict['_r']), 'r': ('b', _dict['l_']),
                  'd': ('d', _dict['r_']), 'l': ('f', _dict['r_'])},
            'l': {'u': ('u', _dict['l_']), 'r': ('f', _dict['l_']),
                  'd': ('d', _dict['_l']), 'l': ('b', _dict['r_'])},
            'u': {'u': ('b', _dict['_u']), 'r': ('r', _dict['_u']),
                  'd': ('f', _dict['u_']), 'l': ('l', _dict['u_'])},
            'd': {'u': ('f', _dict['d_']), 'r': ('r', _dict['d_']),
                  'd': ('b', _dict['_d']), 'l': ('l', _dict['_d'])},
            'b': {'u': ('u', _dict['_u']), 'r': ('l', _dict['l_']),
                  'd': ('d', _dict['_d']), 'l': ('r', _dict['r_'])}
        }

        return tile_map

    # set tile_array associated to target face
    def _set_tile_array(self, target_face):
        # main face elements
        _cube_st = self.cube_state[Cube.face_id[target_face]]
        self.tile_array[1:1+self.order, 1:1+self.order] = _cube_st

        # edge elements
        for edge_id, idx_rng in self.idx_range.items():
            face_id, tile_idx = self.tile_map[target_face][edge_id]
            _cube_st = self._get_edge_tiles(face_id, tile_idx)
            self.tile_array[idx_rng] = _cube_st

    # TODO: could be simplified...
    def _get_edge_tiles(self, face_id, tile_idx):
        face_val = Cube.face_id[face_id]
        edge_tile = [self.cube_state[face_val, _x, _y] for _x, _y in tile_idx]
        return np.array(edge_tile)

    # TODO: could be simplified...
    def _set_edge_tiles(self, face_id, tile_idx, edge_tile):
        face_val = Cube.face_id[face_id]
        for _i, (_x, _y) in enumerate(tile_idx):
            self.cube_state[face_val, _x, _y] = edge_tile[_i]

    # rotate tile_array and update
    # NOTE: inv=False(True) for clockwise(anti-clockwise)
    def _rotate_tile_array(self, inv=False):
        if not inv:
            self.tile_array = np.rot90(self.tile_array, -1)
        else:
            self.tile_array = np.rot90(self.tile_array, +1)

    # update cube state from (operated) tile_array
    # inverse op. of set_tile_array
    def _update_cube_state(self, target_face):
        # main face elements
        _tile_arr = self.tile_array[1:1+self.order, 1:1+self.order]
        self.cube_state[Cube.face_id[target_face]] = _tile_arr

        # edge elements
        for edge_id, idx_rng in self.idx_range.items():
            face_id, tile_idx = self.tile_map[target_face][edge_id]
            _tile_arr = self.tile_array[idx_rng]
            self._set_edge_tiles(face_id, tile_idx, _tile_arr)

    # face rotation
    def _rotate_face(self, target_face, inv=False):
        # set up tile_array from cube_state
        self._set_tile_array(target_face)

        # ratate and update tile_array
        self._rotate_tile_array(inv)

        # set back tile_array to cube state
        self._update_cube_state(target_face)

    # limit action to keep 'static' cube
    def _limit_action(self, face, inv):
        # refer to parent class variable
        if face in type(self).actionable_face:
            return True
        print('WARN: unactionable action ({}) was chosen'.format(face))
        return False

    # NOTE: command: char sequence of operations (ex. 'fl.ru')
    def apply_action(self, command):
        inv = False
        for cmd_char in command:
            if cmd_char is '.':
                inv = not inv
            elif cmd_char in Cube.face_id.keys():
                # limit actions
                if self._limit_action(cmd_char, inv):
                    self._rotate_face(cmd_char, inv)
                inv = False

    # get cube state as flatten vector
    def get_state(self):
        return self.cube_state.flatten()

    # set cube state from flatten vector
    def set_state(self, state_vector):
        self.cube_state = state_vector.reshape(*self.cube_state.shape)

    # solve judgement
    def is_solved(self):
        return (self.cube_state == self.init_cube_state).all()

    # re-initialize state
    def reset_cube(self):
        self.cube_state = self.init_cube_state.copy()

    # std output for debugging
    def display_cube(self):
        order = self.order
        w_char = ' '

        # display top part
        for row in range(order):
            for j in range(order):
                sys.stdout.write(w_char)
            for tile in self.cube_state[Cube.face_id['u'], row]:
                sys.stdout.write(str(tile))
            for j in range(2 * order):
                sys.stdout.write(w_char)
            sys.stdout.write('\n')
        # display middle part
        for row in range(order):
            for tile in self.cube_state[Cube.face_id['l'], row]:
                sys.stdout.write(str(tile))
            for tile in self.cube_state[Cube.face_id['f'], row]:
                sys.stdout.write(str(tile))
            for tile in self.cube_state[Cube.face_id['r'], row]:
                sys.stdout.write(str(tile))
            for tile in self.cube_state[Cube.face_id['b'], row]:
                sys.stdout.write(str(tile))
            sys.stdout.write('\n')
        # display bottom part
        for row in range(order):
            for j in range(order):
                sys.stdout.write(w_char)
            for tile in self.cube_state[Cube.face_id['d'], row]:
                sys.stdout.write(str(tile))
            for j in range(2 * order):
                sys.stdout.write(w_char)
            sys.stdout.write('\n')
        sys.stdout.write('\n')


# 3x3 cube class
class Cube3x3(Cube):
    # edge tile idxs in a face (idx ascend/descend order)
    edge_tile = {
        'u_': [(0, 0), (0, 1), (0, 2)], 'r_': [(0, 2), (1, 2), (2, 2)],
        'd_': [(2, 0), (2, 1), (2, 2)], 'l_': [(0, 0), (1, 0), (2, 0)],
        '_u': [(0, 2), (0, 1), (0, 0)], '_r': [(2, 2), (1, 2), (0, 2)],
        '_d': [(2, 2), (2, 1), (2, 0)], '_l': [(2, 0), (1, 0), (0, 0)]
    }
    # limit actions to keep 'static' cube
    actionable_face = Cube.face_id.keys()

    def __init__(self):
        super().__init__(order=3)


# 2x2 cube class
class Cube2x2(Cube):
    # edge tile idxs in a face (idx ascend/descend order)
    edge_tile = {
        'u_': [(0, 0), (0, 1)], 'r_': [(0, 1), (1, 1)],
        'd_': [(1, 0), (1, 1)], 'l_': [(0, 0), (1, 0)],
        '_u': [(0, 1), (0, 0)], '_r': [(1, 1), (0, 1)],
        '_d': [(1, 1), (1, 0)], '_l': [(1, 0), (0, 0)]
    }
    # limit actions to keep 'static' cube
    actionable_face = ['f', 'r', 'u']

    def __init__(self):
        super().__init__(order=2)


def main():
    cube = Cube3x3()
    cube.display_cube()
    print(cube.is_solved())

    cube.apply_action('f')
    cube.display_cube()

    cube.apply_action('r')
    cube.display_cube()

    cube.apply_action('.u')
    cube.display_cube()

    cube.apply_action('.b')
    cube.display_cube()
    print(cube.is_solved())


if __name__ == "__main__":
    main()
