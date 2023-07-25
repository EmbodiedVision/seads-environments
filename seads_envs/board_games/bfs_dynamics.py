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
BFS dynamics for LightsOut and TileSwap board games (node expansion rules)
"""


class LightsOutBfsDynamics:
    def expand_binary_boards(self, binary_boards):
        """
        Expand `LightsOut` boards by all possible actions

        Parameters
        ----------
        binary_boards: np.ndarray, shape B x N x N
            Batch of binary NxN `LightsOut` boards

        Returns
        -------
        exp_binary_boards: np.ndarray, shape B*n_actions x N x N
            Batch of packed expanded binary boards,
            with n_actions = N*N.
        """
        assert binary_boards.ndim == 3
        boardsize = binary_boards.shape[-1]
        n_actions = boardsize ** 2
        assert binary_boards.shape[-2:] == (boardsize, boardsize)
        assert binary_boards.dtype == bool
        batchsize = binary_boards.shape[0]
        expanded_boards = binary_boards.copy()[:, None, :, :].repeat(n_actions, axis=1)
        for action in range(boardsize ** 2):
            row = action // boardsize
            col = action - boardsize * row
            expanded_boards[:, action, row, col] = ~expanded_boards[:, action, row, col]
            if row >= 1:
                expanded_boards[:, action, row - 1, col] = ~expanded_boards[
                    :, action, row - 1, col
                ]
            if row < boardsize - 1:
                expanded_boards[:, action, row + 1, col] = ~expanded_boards[
                    :, action, row + 1, col
                ]
            if col >= 1:
                expanded_boards[:, action, row, col - 1] = ~expanded_boards[
                    :, action, row, col - 1
                ]
            if col < boardsize - 1:
                expanded_boards[:, action, row, col + 1] = ~expanded_boards[
                    :, action, row, col + 1
                ]
        expanded_boards = expanded_boards.reshape(
            batchsize * n_actions, boardsize, boardsize
        )
        return expanded_boards


class TileSwapBfsDynamics:
    def expand_binary_boards(self, binary_boards):
        """
        Expand `TileSwap` boards by all possible actions.

        Parameters
        ----------
        binary_boards: np.ndarray, shape B x 9 x 9
            Batch of binary `TileSwap` boards.

        Returns
        -------
        exp_binary_boards: np.ndarray, shape B*n_actions x 9 x 9
            Batch of expanded binary boards
        """

        """
        An exemplary binary TileSwap board representation is
        0 1 0 : Chip '1' on tile '0'
        0 0 1 : Chip '2' on tile '1'
        1 0 0 : Chip '0' on tile '2'

        A `TileSwap` action changes chips between two *adjacent* tiles.
        As we have 9 tiles, there are 12 swap operations.
        We manually define which action affects which tiles in TILES_FOR_ACTION.

        Tile indices:
        0 1 2
        3 4 5
        6 7 8

        """
        TILES_FOR_ACTION = [
            # horizontal
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            # vertical
            (0, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 8),
        ]
        assert binary_boards.ndim == 3
        boardsize = 3
        n_actions = len(TILES_FOR_ACTION)
        assert binary_boards.shape[-2:] == (boardsize ** 2, boardsize ** 2)
        assert binary_boards.dtype == bool
        batchsize = binary_boards.shape[0]
        expanded_boards = binary_boards.copy()[:, None, :, :].repeat(n_actions, axis=1)
        for action in range(n_actions):
            swap_pair = TILES_FOR_ACTION[action]
            row_0 = expanded_boards[:, action, swap_pair[0], :].copy()
            expanded_boards[:, action, swap_pair[0], :] = expanded_boards[
                :, action, swap_pair[1], :
            ]
            expanded_boards[:, action, swap_pair[1], :] = row_0
        expanded_boards = expanded_boards.reshape(
            batchsize * n_actions, boardsize ** 2, boardsize ** 2
        )
        return expanded_boards
