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


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow

from seads_envs.board_games.discrete_environments import TileSwapDiscreteEnv
from seads_envs.hybrid.cursor.common import CursorEnv
from seads_envs.hybrid.touch_board import TileSwapTouchBoard

__all__ = ["TileSwapCursorEnv"]


class TileSwapCursorEnv(CursorEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=3,
        random_solution_depth=False,
        mixed_action_space=False,
        done_if_solved=False,
    ):
        """
        Initialize `TileSwapCursorEnv`.

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
        """
        tile_board_env = TileSwapDiscreteEnv(
            reset_split=reset_split,
            board_size=board_size,
            max_solution_depth=max_solution_depth,
            random_solution_depth=random_solution_depth,
        )
        _tile_board = TileSwapTouchBoard(tile_board_env.board_size)
        super().__init__(
            tile_board_env,
            _tile_board,
            mixed_action_space,
            done_if_solved,
        )
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

    def plot_transition(self, ax, obs, action, next_obs):
        cursor = obs[:2]
        print(cursor)
        next_cursor = next_obs[:2]
        action_patch = FancyArrow(*cursor, *(next_cursor - cursor))
        ax.add_patch(action_patch)
        dx, dy, push = self._process_action(action)
        if push:
            target_patch = Circle(next_cursor, radius=0.025, color="r", fill=False)
            ax.add_patch(target_patch)


def test_tileswap():
    fig, ax = plt.subplots(nrows=3, ncols=2)
    env = TileSwapCursorEnv(reset_split="all", max_solution_depth=5)
    cursor_positions = [
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([5 / 6, 1 / 6]),
    ]
    actions = [np.array([0.3, 0, 1]), np.array([0.3, 0.3, 1]), np.array([0, 0.3, 1])]
    for i, (cursor_position, action) in enumerate(zip(cursor_positions, actions)):
        env.reset()
        env.cursor = cursor_position
        obs = env.get_current_observation()
        print(env.board_state)
        env.plot_board(ax[i, 0], env.board_state)
        obs_next, _, _, _ = env.step(action)
        env.plot_transition(ax[i, 0], obs, action, obs_next)
        print(env.board_state)
        env.plot_board(ax[i, 1], env.board_state)

    plt.show()


if __name__ == "__main__":
    test_tileswap()
