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
#
# This source file contains code excerpts from the directory
# https://github.com/deepmind/deepmind-research/tree/7e7255eed10d227154cd746614642d0322ada755/physics_planning_games/,
# licensed under the Apache License, Version 2.0.
# Copyright 2020 DeepMind Technologies Limited.
# The above repository is abbreviated by `<PPG>` in the following.
# Code excerpts taken from this repository are marked case-by-case.

"""
LightsOut logic wrapper for use in manipulation tasks.
"""


class LightsOutGameLogic:
    """Logic for LightsOut game, based on the 'LightsOutDiscreteEnv'"""

    # The structure of this class is inspired by the `TicTacToeGameLogic` class in
    # `<PPG>/board_games/tic_tac_toe_logic.py`.

    def __init__(self, lights_out_discrete):
        self._lights_out_discrete = lights_out_discrete
        self.reset()

    @property
    def board_size(self):
        return self._lights_out_discrete.board_size

    @property
    def max_solution_depth(self):
        return self._lights_out_discrete.max_solution_depth

    def seed(self, seed=None):
        self._lights_out_discrete.seed(seed)

    def reset(self):
        self._lights_out_discrete.reset()

    @property
    def is_game_over(self):
        return False

    # Code block from <PPG>/board_games/tic_tac_toe_logic.py, modified
    def get_board_state(self):
        """Returns the logical board state as a numpy array.

        Returns:
          A boolean array of shape (H, W), where H, W are height and width
          of the board.
        """
        return self._lights_out_discrete.board_state.copy()

    # Code block from <PPG>/board_games/tic_tac_toe_logic.py, modified
    def apply(self, contact_info):
        """Checks whether action is valid, and if so applies it to the game state.

        Args:
          contact_info: A `LightsOutBoardContact` instance.

        Returns:
          True if the action was valid, else False.
        """
        row, col = contact_info.row, contact_info.col
        groundtruth_skill = self._lights_out_discrete.push_field(row, col)
        was_valid_move = True
        return was_valid_move, groundtruth_skill
