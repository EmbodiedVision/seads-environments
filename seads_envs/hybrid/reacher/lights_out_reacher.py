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

from seads_envs.board_games import LightsOutDiscreteEnv
from seads_envs.hybrid.reacher.common import ReacherEnv
from seads_envs.hybrid.touch_board import LightsOutTouchBoard

__all__ = ["LightsOutReacherEnv"]

ON_BRIGHT_RGB = np.array((230 / 255, 58 / 255, 58 / 255))
ON_DARK_RGB = np.array((154 / 255, 38 / 255, 38 / 255)) * 0.85
OFF_BRIGHT_RGB = np.array((230 / 255, 230 / 255, 230 / 255))
OFF_DARK_RGB = np.array((154 / 255, 154 / 255, 154 / 255)) * 0.85


class LightsOutReacherEnv(ReacherEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=5,
        random_solution_depth=False,
        done_if_solved=False,
        mixed_action_space=False,
        action_repeat=2,
        render_width=256,
        render_height=256,
    ):
        """

        Initialize `LightsOutReacherEnv`.

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

        assert board_size == 5

        lights_out_env = LightsOutDiscreteEnv(
            reset_split=reset_split,
            max_solution_depth=max_solution_depth,
            board_size=board_size,
            random_solution_depth=random_solution_depth,
        )
        lightsout_board = LightsOutTouchBoard(lights_out_env.board_size)

        super().__init__(
            lights_out_env,
            lightsout_board,
            mixed_action_space,
            done_if_solved,
            action_repeat,
            render_width,
            render_height,
        )

        self.seed(None)
        self.reset()

    def _register_push(self, pos):
        if np.min(pos) < 0 or np.max(pos) > 1:
            raise ValueError("Invalid position")
        row, col = self.board.get_field_from_pos(pos)
        groundtruth_skill = self._discrete_env.push_field(row, col)
        return groundtruth_skill

    def _set_texture(self):
        rgb_array = np.zeros((self.board_size, self.board_size, 3))
        for i, j in product(range(self.board_size), range(self.board_size)):
            prefix = "on" if self.board_state[i, j] else "off"
            suffix = "dark" if ((i % 2) == (j % 2)) else "bright"
            rgb_value = globals()[(prefix + "_" + suffix + "_rgb").upper()]
            rgb_array[i, j] = 255 * rgb_value
        rgb_array = rgb_array[::-1, :, :].astype(int)
        new_texture = np.repeat(rgb_array, 300 // self.board_size, axis=0)
        new_texture = np.repeat(new_texture, 300 // self.board_size, axis=1)
        self._modder.whiten_materials(["board"])
        self._modder.set_rgb("board", new_texture)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = LightsOutReacherEnv(
        "all",
        max_solution_depth=4,
        board_size=5,
        mixed_action_space=False,
        action_repeat=2,
    )
    env.seed(43)
    env.reset()

    print(env._get_finger_pos())
    plt.imshow(env.render("rgb_array"))
    plt.show()
    env.step(np.array([1, 1, 2]))
    print(env._get_finger_pos())
    plt.imshow(env.render("rgb_array"))
    plt.show()
