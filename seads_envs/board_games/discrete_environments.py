# Copyright 2022 Max-Planck-Gesellschaft
# Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
# Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Discrete environments (board games)
"""

import warnings

import gym
import numpy as np
from gym import spaces

from seads_envs.board_games.board_loader import BoardLoader
from seads_envs.board_games.board_repr import (
    LightsOutBoardRepr,
    TileSwapBoardRepr,
    classify_board_split,
)

__all__ = [
    "LightsOutDiscreteEnv",
    "TileSwapDiscreteEnv",
]


class LightsOutDiscreteEnv(gym.Env):
    """ LightsOut board game """

    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=5,
        random_solution_depth=False,
        reset_by_action_sampling=False,
    ):
        """
        Initialize LightsOut board game.

        Parameters
        ----------
        reset_split: str
            Split of board state of reset env, in ["train", "test"]
        max_solution_depth: int
            (Maximal) solution depth of reset board.
            If `random_solution_depth` is set, the solution depth
            is uniformly sampled from {1, ..., `max_solution_depth`}
        board_size: int
            Board size (one side) (number of fields: board_size ** 2), typically 5 for LightsOut
        random_solution_depth: bool
            Randomly sample solution depth in {1, ..., `max_solution_depth`}
        reset_by_action_sampling: bool
            Instead of using pre-computed boards for initialization,
            generate initial boards by executing randomly sampled actions on an "all lights off" board.
            This may lead to actual solution depths which are smaller than `max_solution_depth`.
            This feature is mainly used for large boards (boardsize > 5), for which pre-computing
            boards of particular solution depths is computationally challenging.
        """
        self.reset_split = reset_split
        self.max_solution_depth = max_solution_depth
        self.board_size = board_size
        self.random_solution_depth = random_solution_depth
        self.action_space = spaces.Discrete(board_size ** 2)
        self.observation_space = spaces.MultiBinary([board_size, board_size])
        self.reset_by_action_sampling = reset_by_action_sampling
        if not reset_by_action_sampling:
            self.board_loader = BoardLoader(
                f"lightsout_boards_s{board_size}.h5py",
                boardsize=board_size,
            )
        self.board_repr = LightsOutBoardRepr(board_size)

        # instance variables
        self.board_state = None
        self.rng = None

        self.seed(None)
        self.reset()

    @property
    def symbolic_shape(self):
        return self.board_size, self.board_size

    @property
    def binary_symbolic_shape(self):
        return self.board_size, self.board_size

    @property
    def symbolic_state(self):
        return self.board_state

    @property
    def binary_symbolic_state(self):
        return self.board_state.astype(bool)

    @property
    def symbolic_target_state(self):
        return np.zeros((self.board_size, self.board_size)).astype(bool)

    @property
    def binary_symbolic_target_state(self):
        return self.symbolic_target_state.astype(bool)

    def set_binary_symbolic_state(self, binary_symbolic_state):
        assert binary_symbolic_state.shape == self.board_state.shape
        self.board_state = binary_symbolic_state.copy()

    def is_solved(self):
        return np.all(self.symbolic_state == self.symbolic_target_state)

    def state_to_binary(self, symbolic_state):
        assert symbolic_state.shape == self.symbolic_shape
        return symbolic_state.astype(bool)

    def binary_to_state(self, binary_state):
        assert binary_state.shape == self.binary_symbolic_shape
        return binary_state

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def _zero_board(self):
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=bool)

    def reset(self, return_action_sequence=False):
        if self.reset_split == "all":
            warnings.warn("reset_split == 'all'")

        if self.random_solution_depth:
            solution_depth = self.rng.randint(1, self.max_solution_depth + 1)
        else:
            solution_depth = self.max_solution_depth

        if return_action_sequence:
            raise NotImplementedError

        if self.reset_by_action_sampling:
            matches_split = False
            while not matches_split:
                self._zero_board()
                action_seq = np.arange(self.board_size ** 2)
                self.rng.shuffle(action_seq)
                for action in action_seq[:solution_depth]:
                    self.apply_action_to_board_inplace(self.board_state, action)
                if self.reset_split == "all":
                    matches_split = True
                else:
                    split = classify_board_split(self.board_state)
                    matches_split = split == self.reset_split
        else:
            packed_board = self.board_loader.load_packed_board(
                solution_depth, self.reset_split, self.rng
            )
            board = self.board_repr.packed_to_canonical(packed_board)
            self.board_state = np.copy(board)

        return np.copy(self.board_state)

    def _push_field_on_board_inplace(self, board_state, row, col):
        board_state[row, col] = ~board_state[row, col]
        if row >= 1:
            board_state[row - 1, col] = ~board_state[row - 1, col]
        if row < self.board_size - 1:
            board_state[row + 1, col] = ~board_state[row + 1, col]
        if col >= 1:
            board_state[row, col - 1] = ~board_state[row, col - 1]
        if col < self.board_size - 1:
            board_state[row, col + 1] = ~board_state[row, col + 1]

    def apply_action_to_board_inplace(self, board_state, action):
        board_size = self.board_size
        row = action // board_size
        col = action - board_size * row
        self._push_field_on_board_inplace(board_state, row, col)

    def push_field(self, row, col):
        self._push_field_on_board_inplace(self.board_state, row, col)
        return row * self.board_size + col

    def step(self, action: int):
        row = action // self.board_size
        col = action - self.board_size * row
        self.push_field(row, col)
        obs, reward, done, info = np.copy(self.board_state), 0, False, {}
        return obs, reward, done, info

    def render(self, mode="human"):
        raise NotImplementedError


class TileSwapDiscreteEnv(gym.Env):
    """ TileSwap board game """

    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=3,
        random_solution_depth=False,
    ):
        """
        Initialize TileSwap board game.

        Parameters
        ----------
        reset_split: str
            Split of board state of reset env, in ["train", "test"]
        max_solution_depth: int
            (Maximal) solution depth of reset board.
            If `random_solution_depth` is set, the solution depth
            is uniformly sampled from {1, ..., `max_solution_depth`}
        board_size: int
            Board size (one side) (number of fields: board_size ** 2), typically 3 for TileSwap
        random_solution_depth: bool
            If True, randomly (uniformly) sample solution depth in {1, ..., `max_solution_depth`}
        """
        self.reset_split = reset_split
        self.max_solution_depth = max_solution_depth
        self.board_size = board_size
        self.random_solution_depth = random_solution_depth
        self.observation_space = None
        self.ORDERED_BOARDSTATE = np.arange(self.board_size ** 2).reshape(
            (self.board_size, self.board_size)
        )
        n_actions = 2 * board_size * (board_size - 1)
        self.action_space = spaces.Discrete(n_actions)

        self.board_loader = BoardLoader(
            f"tileswap_boards_s{board_size}.h5py",
            boardsize=board_size,
        )
        self.board_repr = TileSwapBoardRepr(board_size)

        self.board_state = None
        self.rng = None

        self.seed(None)
        self.reset()

    @property
    def symbolic_shape(self):
        return self.board_size, self.board_size

    @property
    def binary_symbolic_shape(self):
        return self.board_size ** 2, self.board_size ** 2

    @property
    def symbolic_state(self):
        return self.board_state.copy()

    @property
    def binary_symbolic_state(self):
        return self.state_to_binary(self.symbolic_state)

    @property
    def symbolic_target_state(self):
        return self.ORDERED_BOARDSTATE.copy()

    @property
    def binary_symbolic_target_state(self):
        return self.state_to_binary(self.symbolic_target_state)

    def is_solved(self):
        return np.all(self.symbolic_state == self.symbolic_target_state)

    def state_to_binary(self, symbolic_state):
        assert symbolic_state.shape == self.symbolic_shape
        board_state_flat = symbolic_state.reshape(-1)
        board_state_onehot = np.eye(self.board_size ** 2)[board_state_flat]
        return board_state_onehot.astype(bool)

    def binary_to_state(self, binary_state):
        assert binary_state.shape == self.binary_symbolic_shape
        board_state = np.zeros(self.board_size ** 2).astype(int)
        for field_idx in range(self.board_size ** 2):
            nz_at = np.nonzero(binary_state[field_idx])[0]
            board_state[field_idx] = nz_at[0]
        board_state = board_state.reshape(self.board_size, self.board_size)
        return board_state

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def _zero_board(self):
        self.board_state = np.copy(self.ORDERED_BOARDSTATE)

    def reset(self, return_action_sequence=False):
        if self.reset_split == "all":
            warnings.warn("reset_split == 'all'")

        if self.random_solution_depth:
            solution_depth = self.rng.randint(1, self.max_solution_depth + 1)
        else:
            solution_depth = self.max_solution_depth

        if return_action_sequence:
            raise NotImplementedError

        packed_board = self.board_loader.load_packed_board(
            solution_depth, self.reset_split, self.rng
        )
        board = self.board_repr.packed_to_canonical(packed_board)
        self.board_state = np.copy(board)
        return np.copy(self.board_state)

    def _get_obs(self):
        return np.copy(self.board_state)

    def _swap_inplace(self, index_a, index_b, board_state):
        assert board_state.shape[-1] == self.board_size
        assert board_state.shape[-2] == self.board_size
        index_a = tuple(index_a)
        index_b = tuple(index_b)
        board_state[index_a], board_state[index_b] = (
            board_state[index_b],
            board_state[index_a],
        )

    def _get_swap_indices(self, action):
        if action < self.board_size * (self.board_size - 1):  # Horizontal swap
            index_left = (
                action % (self.board_size - 1),
                action // (self.board_size - 1),
            )  # Index of left element
            swap_indices = (index_left, (index_left[0] + 1, index_left[1]))
        else:  # Vertical swap
            index_upper = (
                (action - self.board_size * (self.board_size - 1)) % self.board_size,
                (action - self.board_size * (self.board_size - 1)) // self.board_size,
            )  # Index of upper element
            swap_indices = (index_upper, (index_upper[0], index_upper[1] + 1))
        return swap_indices

    def _get_action(self, swap_indices):
        index_a, index_b = swap_indices
        if index_a[1] == index_b[1]:
            # Horizontal swap
            index_left_x = min(index_a[0], index_b[0])
            index_left_y = index_a[1]
            action = index_left_y * (self.board_size - 1) + index_left_x
        elif index_a[0] == index_b[0]:
            # Vertical swap
            index_upper_x = index_a[0]
            index_upper_y = min(index_a[1], index_b[1])
            n_actions = self.board_size * (self.board_size - 1)
            action = n_actions + (index_upper_y * self.board_size + index_upper_x)
        else:
            raise ValueError
        return action

    def step(self, action: int):
        swap_indices = self._get_swap_indices(int(action))
        self.swap(*swap_indices)
        obs, reward, done, info = self._get_obs(), 0, False, {}
        return obs, reward, done, info

    def _simulate_step(self, candidate_action):
        swap_indices = self._get_swap_indices(int(candidate_action))
        new_board = self.board_state.copy()
        self._swap_inplace(*swap_indices, new_board)
        return new_board

    def swap(self, index_a, index_b):
        # In addition to super().swap() return groundtruth skill index.
        index_a = tuple(index_a)
        index_b = tuple(index_b)
        self._swap_inplace(index_a, index_b, self.board_state)
        action = self._get_action((index_a, index_b))
        return action

    def apply_action_to_board_inplace(self, board_state, action):
        index_blank, index_tile = self._get_swap_indices(action)
        self._swap_inplace(index_blank, index_tile, board_state)

    def render(self, mode="human"):
        raise NotImplementedError


def test_lights_out_discrete_env():
    env = LightsOutDiscreteEnv(
        reset_split="train",
        max_solution_depth=1,
        board_size=5,
        reset_by_action_sampling=True,
    )
    env.seed(42)
    obs_init = env.reset()
    print(obs_init)
    obs_init = env.reset()
    print(obs_init)
    obs_next, _, _, _ = env.step(env.action_space.sample())
    print(obs_next)


def test_tile_swap_discrete_env():
    env = TileSwapDiscreteEnv(
        reset_split="train",
        max_solution_depth=1,
        board_size=3,
        random_solution_depth=False,
    )
    env.seed(42)
    obs_init = env.reset()
    print(obs_init)
    obs_init = env.reset()
    print(obs_init)
    obs_next, _, _, _ = env.step(env.action_space.sample())
    print(obs_next)


if __name__ == "__main__":
    test_lights_out_discrete_env()
    test_tile_swap_discrete_env()
