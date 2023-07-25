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

from itertools import product

import numpy as np

from seads_envs.board_games.board_renderer import (
    LightsOutBoardRenderer,
    TileSwapBoardRenderer,
)

__all__ = ["LightsOutTouchBoard", "LightsOutSpacedTouchBoard", "TileSwapTouchBoard"]


class TouchBoard:
    def __init__(self, board_size):
        self.board_size = board_size

    def render_board(self, board_state):
        raise NotImplementedError

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        if board_state is None:
            board_state = np.zeros((self.board_size, self.board_size))
        rgb_array = self.render_board(board_state)
        ax.imshow(rgb_array, extent=[0, 1, 1, 0])
        ax.set_xticks([])
        ax.set_yticks([])


class LightsOutTouchBoard(TouchBoard):
    def __init__(self, board_size):
        super(LightsOutTouchBoard, self).__init__(board_size)
        self.right_boundaries = np.linspace(0, 1, board_size + 1)[1:]
        self.board_renderer = LightsOutBoardRenderer(board_size)

    def get_field_from_pos(self, pos):
        if np.min(pos) < 0 or np.max(pos) > 1:
            raise ValueError("Invalid position")
        row = np.digitize(pos[1], self.right_boundaries, right=True)
        col = np.digitize(pos[0], self.right_boundaries, right=True)
        return row, col

    def get_vertical_halffield(self, pos):
        pos = np.clip(pos, 0, 0.999)
        y = pos[1]
        # 0 is upper half of first field, 1 lower half, 2 upper half of sec. field, ...
        halffield = int(y * 2 * self.board_size)
        halffield_name = "upper" if halffield % 2 == 0 else "lower"
        return halffield_name

    def render_board(self, board_state):
        return self.board_renderer.render_board(board_state)


class LightsOutSpacedTouchBoard(TouchBoard):
    def __init__(self, board_size, relative_spacing):
        super(LightsOutSpacedTouchBoard, self).__init__(board_size)
        self.relative_spacing = relative_spacing
        assert 5 <= board_size <= 13
        assert board_size % 2 == 1
        # each tile is of width/height 1/13
        remaining_space_total = 1 - board_size * (1 / 13)
        space_between_tiles = relative_spacing * (
            remaining_space_total / (board_size - 1)
        )
        distance_centerpoints = 1 / 13 + space_between_tiles
        centerpoints = (
            0.5
            + np.linspace(-(board_size - 1) / 2, (board_size - 1) / 2, board_size)
            * distance_centerpoints
        )
        self.Xc, self.Yc = np.meshgrid(centerpoints, centerpoints)
        self.tile_width = 1 / 13

    def get_field_from_pos(self, pos):
        if np.min(pos) < 0 or np.max(pos) > 1:
            raise ValueError("Invalid position")
        dist_x = np.abs(pos[0] - self.Xc)
        dist_y = np.abs(pos[1] - self.Yc)
        match_x = dist_x < self.tile_width / 2
        match_y = dist_y < self.tile_width / 2
        match = np.logical_and(match_x, match_y)
        if np.sum(match) == 0:
            row = None
            col = None
        elif np.sum(match) == 1:
            row, col = np.argwhere(match)[0]
        else:
            raise RuntimeError
        return row, col

    def render_board(self, board_state):
        board_size = board_state.shape[0]
        render_width = 13 * 21
        render_height = 13 * 21
        on_bright_rgb = np.array((230, 58, 58))
        off_bright_rgb = np.array((230, 230, 230))
        rgb_array = 255 * np.ones((render_height, render_width, 3)).astype(np.uint8)

        for i, j in product(range(board_size), range(board_size)):
            row_center = int(self.Yc[i, j] * render_height)
            col_center = int(self.Xc[i, j] * render_width)
            color = on_bright_rgb if board_state[i, j] else off_bright_rgb
            rgb_array[
                max(0, row_center - 11) : row_center + 11,
                max(0, col_center - 11) : col_center + 11,
                :,
            ] = 0
            rgb_array[
                max(0, row_center - 10) : row_center + 10,
                max(0, col_center - 10) : col_center + 10,
                :,
            ] = color

        return rgb_array


class TileSwapTouchBoard(TouchBoard):
    def __init__(self, board_size):
        super(TileSwapTouchBoard, self).__init__(board_size)
        self._pivot_points, self._swap_indices = self._compute_pivot_points(board_size)
        self.board_renderer = TileSwapBoardRenderer(board_size, self._pivot_points)

    def _compute_pivot_points(self, board_size):
        """
        Compute coordinates of pivot points on board with normalized extents of [0, 1].
        The origin (0, 0) of cursor is at topleft of the board.
        The field (0, 0) is the topleft field of the board.

        Returns
        -------
        pivot_points: list
            List of pivot points
        swap_indices: list
            List of swap indices
        """
        board_size = board_size
        pivot_points = []
        swap_indices = []
        fieldsize = 1 / board_size
        for row in range(board_size):
            for col in range(board_size):
                # pivot points on horizontal lines
                if row != board_size - 1:
                    pp_x = 0.5 * fieldsize + fieldsize * col
                    pp_y = (row + 1) * fieldsize
                    pivot_points.append([pp_x, pp_y])
                    swap_indices.append(((row, col), (row + 1, col)))
                # pivot points on vertical lines
                if col != board_size - 1:
                    pp_x = fieldsize * (col + 1)
                    pp_y = 0.5 * fieldsize + row * fieldsize
                    pivot_points.append([pp_x, pp_y])
                    swap_indices.append(((row, col), (row, col + 1)))
        return pivot_points, swap_indices

    def get_swap_pair_from_pos(self, pos):
        """
        Compute indices of fields to be swapped from normalized position.

        Parameters
        ----------
        pos: np.ndarray
            Position on board, in [0, 1]^2

        Returns
        -------
        swap_pair: tuple(tuple(int, int), tuple(int, int))
            Pair of fields (row, col) to be swapped
        """
        if np.min(pos) < 0 or np.max(pos) > 1:
            raise ValueError("Invalid position")
        dist = (np.abs((self._pivot_points - pos[None, :]))).sum(axis=-1)
        fieldsize = 1 / self.board_size
        min_dist_idx = np.argmin(dist)
        min_dist = dist[min_dist_idx]
        if min_dist <= fieldsize / 2:
            return self._swap_indices[min_dist_idx]
        else:
            return None

    def render_board(self, board_state):
        raise NotImplementedError

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        return self.board_renderer.plot_board(
            ax, board_state=board_state, initial_board_state=initial_board_state
        )
