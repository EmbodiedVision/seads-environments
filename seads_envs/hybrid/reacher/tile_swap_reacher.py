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
TileSwapReacher environment
"""

from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from seads_envs import ASSETS_DIR
from seads_envs.board_games import TileSwapDiscreteEnv
from seads_envs.hybrid.reacher.common import ReacherEnv
from seads_envs.hybrid.touch_board import TileSwapTouchBoard

__all__ = ["TileSwapReacherEnv"]


class TileSwapReacherEnv(ReacherEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=3,
        random_solution_depth=False,
        done_if_solved=False,
        mixed_action_space=False,
        action_repeat=2,
        render_width=256,
        render_height=256,
    ):
        """
        Initialize `TileSwapReacherEnv`.

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
        done_if_solved: bool
            If True, emit 'done' signal only if board is fully solved (i.e., all fields off).
            If False, 'done' is emitted after a field has been pushed.
        mixed_action_space: bool
            If True, use a mixed discrete/continuous action space
        action_repeat: int
            Number of simulation steps that should be taken in mujoco
        render_width: int
            Width (px) of rendering
        render_height: int
            Height (px) of rendering
        """

        assert board_size == 3

        tile_board_env = TileSwapDiscreteEnv(
            reset_split=reset_split,
            board_size=board_size,
            max_solution_depth=max_solution_depth,
            random_solution_depth=random_solution_depth,
        )
        tile_board = TileSwapTouchBoard(tile_board_env.board_size)

        super().__init__(
            tile_board_env,
            tile_board,
            mixed_action_space,
            done_if_solved,
            action_repeat,
            render_width,
            render_height,
        )

        self._tile_textures = None

        self.seed(None)
        self.reset()

    def _register_push(self, pos):
        if np.min(pos) < 0 or np.max(pos) > 1:
            raise ValueError("Invalid position")
        swap_indices = self.board.get_swap_pair_from_pos(pos)
        if swap_indices is None:
            return None
        else:
            groundtruth_skill = self._discrete_env.swap(*swap_indices)
            return groundtruth_skill

    def _set_texture(self):
        if self._tile_textures is None:
            tile_textures = []
            for i in range(10):
                img = Image.open(ASSETS_DIR.joinpath("tileswap", f"{i}.png"))
                img.load()
                tile_textures.append(np.asarray(img)[..., :3])
            self._tile_textures = tile_textures

        tile_texture_size = self._tile_textures[0].shape[0]
        whole_texture_size = self.board_size * tile_texture_size
        new_texture = np.zeros((whole_texture_size, whole_texture_size, 3))
        for i, j in product(range(self.board_size), range(self.board_size)):
            new_texture[
                i * tile_texture_size : (i + 1) * tile_texture_size,
                j * tile_texture_size : (j + 1) * tile_texture_size,
            ] = self._tile_textures[self.board_state[i, j]]

        new_texture = new_texture[::-1]
        self._modder.whiten_materials(["board"])
        self._modder.set_rgb("board", new_texture)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = TileSwapReacherEnv(
        reset_split="all",
        max_solution_depth=4,
        board_size=3,
        mixed_action_space=False,
    )
    env.seed(43)
    env.reset()

    print(env.board_state)
    print(env._get_finger_pos())
    plt.imshow(env.render("rgb_array"))
    plt.show()
    env.step(np.array([1, 1, 2]))
    print(env._get_finger_pos())
    plt.imshow(env.render("rgb_array"))
    plt.show()
