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


import numpy as np

from seads_envs.board_games.discrete_environments import LightsOutDiscreteEnv
from seads_envs.hybrid.cursor.common import CursorEnv
from seads_envs.hybrid.touch_board import LightsOutSpacedTouchBoard, LightsOutTouchBoard

__all__ = ["LightsOutCursorEnv", "LightsOutSpacedCursorEnv"]


class LightsOutCursorEnv(CursorEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=5,
        random_solution_depth=False,
        done_if_solved=False,
        mixed_action_space=False,
        toggle_by_halffield=False,
    ):
        """
        Initialize `LightsOutCursorEnv`.

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
        toggle_by_halffield: bool
            Fields can only be switched *on* when pushing on the upper half,
            and only be switched off when pushing on the lower half
        """
        _lights_out_env = LightsOutDiscreteEnv(
            reset_split,
            max_solution_depth,
            board_size=board_size,
            random_solution_depth=random_solution_depth,
        )
        _lights_out_board = LightsOutTouchBoard(board_size)
        self.toggle_by_halffield = toggle_by_halffield
        super().__init__(
            _lights_out_env,
            _lights_out_board,
            mixed_action_space,
            done_if_solved,
        )

        self.seed(None)
        self.reset()

    def _register_push(self, pos):
        if self.toggle_by_halffield:
            row, col = self.board.get_field_from_pos(pos)
            halffield = self.board.get_vertical_halffield(pos)
            if ~self.board_state[row, col] and halffield == "upper":
                groundtruth_skill = self._discrete_env.push_field(row, col)
            elif self.board_state[row, col] and halffield == "lower":
                groundtruth_skill = self._discrete_env.push_field(row, col)
            else:
                groundtruth_skill = None
        else:
            row, col = self.board.get_field_from_pos(pos)
            groundtruth_skill = self._discrete_env.push_field(row, col)
        return groundtruth_skill


class LightsOutSpacedCursorEnv(CursorEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=5,
        relative_spacing=0,
        random_solution_depth=False,
        done_if_solved=False,
        mixed_action_space=False,
    ):
        """
        Initialize `LightsOutSpacedCursorEnv`.

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
        """
        _lights_out_env = LightsOutDiscreteEnv(
            reset_split,
            max_solution_depth,
            board_size=board_size,
            random_solution_depth=random_solution_depth,
            reset_by_action_sampling=board_size > 5,
        )
        _lights_out_board = LightsOutSpacedTouchBoard(board_size, relative_spacing)
        super().__init__(
            _lights_out_env,
            _lights_out_board,
            mixed_action_space,
            done_if_solved,
        )

        self.seed(None)
        self.reset()

    def _register_push(self, pos):
        row, col = self.board.get_field_from_pos(pos)
        if row is not None:
            groundtruth_skill = self._discrete_env.push_field(row, col)
        else:
            groundtruth_skill = None
        return groundtruth_skill


if __name__ == "__main__":
    env = LightsOutCursorEnv(
        reset_split="train", max_solution_depth=5, toggle_by_halffield=True
    )
    env.seed(42)
    print(env.reset()[2:].reshape(5, 5))
    env.cursor = np.zeros(2)
    obs, _, d, _ = env.step((0, 0.16, 1))
    print(obs[2:].reshape(5, 5))
    print(d)
    obs, _, d, _ = env.step((0, -0.15, 1))
    print(obs[2:].reshape(5, 5))
    print(d)
