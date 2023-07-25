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
Render boards
"""
from itertools import product

import numpy as np
from matplotlib.collections import LineCollection

from seads_envs import ASSETS_DIR

__all__ = [
    "LightsOutBoardRenderer",
    "TileSwapBoardRenderer",
]


class LightsOutBoardRenderer:
    def __init__(self, board_size):
        self._board_size = board_size

    def render_board(self, board_state):
        board_size = board_state.shape[0]
        render_width = 120
        render_height = 120
        on_bright_rgb = np.array((230, 58, 58))
        off_bright_rgb = np.array((230, 230, 230))
        rgb_array = np.zeros(board_state.shape + (3,)).astype(np.uint8)
        for i, j in product(range(board_size), range(board_size)):
            rgb_array[i, j, :] = on_bright_rgb if board_state[i, j] else off_bright_rgb

        rgb_array = np.repeat(rgb_array, render_height // board_size, axis=0)
        rgb_array = np.repeat(rgb_array, render_width // board_size, axis=1)
        assert tuple(rgb_array.shape[:2]) == (render_height, render_width)

        for i in range(board_size + 1):
            idx = 1 + int((i / board_size) * (render_width - 2))
            rgb_array[idx - 1 : idx + 1, :, :] = 0
            rgb_array[:, idx - 1 : idx + 1, :] = 0

        return rgb_array

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        if board_state is None:
            board_state = np.zeros((self._board_size, self._board_size))
        rgb_array = self.render_board(board_state)
        ax.imshow(rgb_array, extent=[0, 1, 1, 0])
        # ax.imshow(255 * np.ones_like(rgb_array), extent=[0, 1, 1, 0], alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])


class TileSwapBoardRenderer:
    def __init__(self, board_size, pivot_points):
        self._board_size = board_size
        self._pivot_points = pivot_points

    def render_board(self):
        raise NotImplementedError

    def plot_board(self, ax, board_state=None, initial_board_state=True):
        tile_px = 100
        assets_path = ASSETS_DIR.joinpath("tileswap")
        from PIL import Image

        if board_state is not None:
            rendering = Image.new(
                "RGB", (tile_px * self._board_size, tile_px * self._board_size)
            )
            for row in range(self._board_size):
                for col in range(self._board_size):
                    tile_asset_path = assets_path.joinpath(
                        f"{board_state[row, col]}.png"
                    )
                    tile = Image.open(tile_asset_path)
                    rendering.paste(tile, (col * tile_px, row * tile_px))
            rendering = np.array(rendering)
            ax.imshow(rendering, extent=[0, 1, 1, 0])

        if self._pivot_points is not None:
            half_field_size = 0.5 / self._board_size
            x_dir = np.array([half_field_size, 0])
            y_dir = np.array([0, half_field_size])
            for pivot in self._pivot_points:
                pivot = np.array(pivot)
                # We compute vertices of the l^1 ball around pivot, walking according to usual convention
                vertices = [pivot + x_dir, pivot + y_dir, pivot - x_dir, pivot - y_dir]
                lines = [(vertices[i % 4], vertices[(i + 1) % 4]) for i in range(0, 4)]
                lc = LineCollection(
                    lines,
                    colors="orange",
                    linewidths=max(1, get_ax_size(ax)[0] / 100),
                    alpha=0.5,
                )
                ax.add_collection(lc)
        ax.set_xticks([])
        ax.set_yticks([])


def get_ax_size(ax):
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height
