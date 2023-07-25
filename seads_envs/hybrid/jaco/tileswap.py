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
TileSwapJaco environment
"""

from itertools import product

import numpy as np
from dm_control import composer

from seads_envs.board_games import TileSwapDiscreteEnv
from seads_envs.hybrid.jaco import composed_task, tileswap_board, tileswap_logic
from seads_envs.hybrid.touch_board import TileSwapTouchBoard
from seads_envs.hybrid.wrapped_discrete import WrappedDiscreteEnv
from seads_envs.third_party.physics_planning_games.board_games.internal import (
    observations,
)
from seads_envs.third_party.physics_planning_games.board_games.internal.observations import (
    _DISABLED_CAMERA,
    _ENABLED_FEATURE,
    _ENABLED_FTT,
    ObservationSettings,
)
from seads_envs.utils.dmc2gym import DMCWrapper

EMPTY_OBSERVABLES = observations.ObservableNames(prop_pose=["position"])

PROPRIO_FEATURES = ObservationSettings(
    proprio=_ENABLED_FEATURE,
    ftt=_ENABLED_FTT,
    prop_pose=_ENABLED_FEATURE,
    board_state=_ENABLED_FEATURE,
    camera=_DISABLED_CAMERA,
)


class TileSwapJacoEnv(WrappedDiscreteEnv):
    def __init__(
        self,
        reset_split,
        max_solution_depth,
        board_size=3,
        random_solution_depth=False,
        done_if_solved=False,
        render_height=256,
        render_width=256,
        control_timestep=0.1,
        rendering_enabled=False,
    ):
        """
        Initialize `TileSwapJacoEnv`

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
        render_width: int
            Width (px) of rendering
        render_height: int
            Height (px) of rendering
        control_timestep: float, optional
            Timestep of internal simulation. Default: 0.1.
        rendering_enabled: bool
            Enable rendering (disable when not needed, this makes the 'step' call significantly
            slower as the rendering is prepared in any step)
        """

        manipulator = composed_task.JacoManipulator(PROPRIO_FEATURES)
        self.board = tileswap_board.TileSwapBoard(board_size)

        self._discrete_env = TileSwapDiscreteEnv(
            reset_split=reset_split,
            max_solution_depth=max_solution_depth,
            board_size=board_size,
            random_solution_depth=random_solution_depth,
        )

        game_logic = tileswap_logic.TileSwapGameLogic(self._discrete_env)
        self._task = composed_task.ManipulatorBoardTask(
            manipulator,
            self.board,
            game_logic,
            observation_settings=PROPRIO_FEATURES,
            control_timestep=control_timestep,
            reset_arm_after_move=True,
            rendering_enabled=rendering_enabled,
        )

        self.done_if_solved = done_if_solved
        self.render_width = render_width
        self.render_height = render_height
        self.time_limit = float("inf")
        self.strip_singleton_obs_buffer_dim = False
        self.rendering_enabled = rendering_enabled

        self._manip_rng = None
        self._dm_env = None
        self._gym_env = None
        # seed env here to make action_space available via self._gym_env
        self.seed(None)
        super().__init__(
            self._discrete_env,
            manipulator_action_space=self._gym_env.action_space,
            manipulator_observation_space=self._gym_env.observation_space,
        )

    def _get_manipulator_observation(self):
        dm_obs = self._dm_env._observation_updater.get_observation()
        obs = self._gym_env._get_obs(dm_obs)
        return obs

    def seed(self, seed=None):
        self._manip_rng = np.random.RandomState(seed)
        self._dm_env = composer.Environment(
            task=self._task,
            time_limit=self.time_limit,
            strip_singleton_obs_buffer_dim=self.strip_singleton_obs_buffer_dim,
        )
        self._gym_env = DMCWrapper(
            self._dm_env,
            height=self.render_height,
            width=self.render_width,
            channels_first=False,
        )
        # resetting the gym env is very expensive.
        # fully reset once here, then only do a "lean reset" in reset()
        self._gym_env.reset()
        # seed the discrete env here to make sure that the next call
        # to reset() resets discrete_env starting with seed 'seed'
        self.discrete_env.seed(seed)

    def reset(self):
        physics = self._dm_env._physics
        with physics.reset_context():
            self._task.initialize_episode(physics, self._manip_rng)
        self._dm_env._observation_updater.reset(physics, self._manip_rng)
        self._task.maybe_update_board_appearance(physics)
        return super().get_current_observation()

    def step(self, action):
        assert self._gym_env._norm_action_space.contains(action)
        action = self._gym_env._convert_action(action)
        assert self._gym_env._true_action_space.contains(action)
        time_step = self._gym_env._env.step(action)
        info = {
            "contact_info": self._task._contact_info,
        }
        if self._task._made_move_this_step:
            info["groundtruth_skill"] = self._task._groundtruth_skill
        if self.done_if_solved:
            done = self.is_solved()
            reward = 1 if done else 0
            info["is_success"] = done
        else:
            reward = 0
            done = self._task._made_move_this_step

        manip_obs = self._gym_env._get_obs(time_step.observation)
        obs = self.get_current_observation(manip_obs)
        return obs, reward, done, info

    def render(self, mode="human"):
        if not self.rendering_enabled:
            raise RuntimeError("Rendering disabled")
        return self._gym_env.render(mode=mode)

    def close(self):
        self._gym_env.close()

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        board = TileSwapTouchBoard(self.board_size)
        return board.plot_board(ax)

    def plot_trajectory(
        self,
        episode,
        ax,
        plot_done=True,
        scatter_kwargs=None,
    ):
        states = [t["state"] for t in episode] + [episode[-1]["next_state"]]
        states = np.stack(states)
        # observation vector:
        # joints_pos 6x2
        # joints_torque 6
        # joints_vel 6
        # hand/joints_pos 3
        # hand/joints_vel 3
        # hand/pinch_site_pos 3
        # hand/pinch_site_rmat 9
        pinch_site_pos_xy = states[:, 30:32]
        board_extent = np.array(
            self.board.board_extent
        )  # extent as ((x1, y1), (x2, y2))
        pinch_site_pos_norm = (pinch_site_pos_xy - board_extent[0][None, :]) / (
            board_extent[1] - board_extent[0]
        )[None, :]
        pinch_site_pos_norm[:, 1] = 1 - pinch_site_pos_norm[:, 1]
        ax.scatter(
            pinch_site_pos_norm[0, 0],
            pinch_site_pos_norm[0, 1],
            alpha=0.5,
            marker="o",
            s=10,
        )
        ax.plot(pinch_site_pos_norm[:, 0], pinch_site_pos_norm[:, 1], alpha=0.5)
        if (
            "contact_info" in episode[-1]["info"]
            and episode[-1]["info"]["contact_info"] is not None
        ):
            norm_pos = episode[-1]["info"]["contact_info"].norm_pos
            ax.scatter(norm_pos[0], norm_pos[1], color="green", marker="x", zorder=10)

    def plot_skill_episodes(
        self,
        episode_list,
        num_skills=None,
        subplots_kwargs=None,
        return_ax=False,
        ax_arr=None,
        plot_done=True,
        scatter_kwargs=None,
    ):
        import matplotlib.pyplot as plt

        if scatter_kwargs is None:
            scatter_kwargs = {}

        n_rows, n_cols = 2 * (self.board_size - 1), self.board_size
        if subplots_kwargs is None:
            subplots_kwargs = {"figsize": (2 * n_cols, 2 * n_rows)}

        if ax_arr is None:
            fig, ax_arr = plt.subplots(nrows=n_rows, ncols=n_cols, **subplots_kwargs)
        else:
            fig = None

        for ax in ax_arr.ravel():
            self.plot_board(ax)

        for row_idx, col_idx in product(range(n_rows), range(n_cols)):
            symbolic_action = row_idx * n_cols + col_idx
            filtered_episodes = list(
                filter(
                    lambda e: (e[0]["symbolic_action"] == symbolic_action)
                    and (e[-1]["info"]["contact_info"] is not None),
                    episode_list,
                )
            )

            if len(filtered_episodes) == 0:
                continue
            try:
                contact_pos = [
                    e[-1]["info"]["contact_info"].norm_pos for e in filtered_episodes
                ]
                contact_pos_arr = np.stack(contact_pos)
                ax_arr[row_idx, col_idx].scatter(
                    contact_pos_arr[:, 0], contact_pos_arr[:, 1]
                )
            except Exception as e:
                print("Scatter plot for contacts failed")

        return (fig, ax_arr) if return_ax else fig
