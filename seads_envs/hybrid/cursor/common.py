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
from gym import spaces

from seads_envs.hybrid.plotting import plot_2d_trajectory, plot_skill_episodes
from seads_envs.hybrid.wrapped_discrete import WrappedDiscreteEnv

__all__ = ["CursorEnv"]


class CursorEnv(WrappedDiscreteEnv):
    """
    General 'Cursor' env.

    The 'cursor' is constrained to the space [0, 1]^2,
    where the origin (0, 0) is in the upper left corner of the array.
    """

    def __init__(self, discrete_env, touch_board, mixed_action_space, done_if_solved):
        dx = dy = 0.2
        if mixed_action_space:
            continuous_space = spaces.Box(
                low=np.array([-dx, -dy]).astype(np.float32),
                high=np.array([dx, dy]).astype(np.float32),
                dtype=np.float32,
            )
            discrete_space = spaces.Discrete(n=2)
            action_space = spaces.Tuple(
                spaces=(continuous_space, discrete_space)
            )  # dx, dy, PUSH
        else:
            action_space = spaces.Box(
                low=np.array([-dx, -dy, -1]).astype(np.float32),
                high=np.array([dx, dy, 1]).astype(np.float32),
                dtype=np.float32,
            )  # dx, dy, PUSH (>PUSH_THRESHOLD)
            self.PUSH_THRESHOLD = -1 + 0.8 * 2

        super().__init__(
            discrete_env,
            manipulator_action_space=action_space,
            manipulator_observation_space=spaces.Box(
                low=np.zeros((2,)).astype(np.float32),
                high=np.ones((2,)).astype(np.float32),
                dtype=np.float32,
            ),
        )

        self.board = touch_board
        self.mixed_action_space = mixed_action_space
        self.done_if_solved = done_if_solved
        self.cursor = None

    def reset(self):
        self.discrete_env.reset()
        self.cursor = self.rng.rand(2)
        return self.get_current_observation()

    def _get_manipulator_observation(self):
        return np.copy(self.cursor)

    def _process_action(self, action):
        if self.mixed_action_space:
            (dx, dy), push = action
        else:
            dx, dy, push_real = action
            push = push_real > self.PUSH_THRESHOLD

        dx = np.clip(dx, self.action_space.low[0], self.action_space.high[0])
        dy = np.clip(dy, self.action_space.low[1], self.action_space.high[1])
        return dx, dy, push

    def _register_push(self, pos):
        raise NotImplementedError

    def step(self, action):
        dx, dy, push = self._process_action(action)
        self.cursor[0] = np.clip(self.cursor[0] + dx, 0, 1)
        self.cursor[1] = np.clip(self.cursor[1] + dy, 0, 1)
        info = {}
        symbolic_state_changed = False

        if push:
            groundtruth_skill = self._register_push(self.cursor)
            info["groundtruth_skill"] = groundtruth_skill
            if groundtruth_skill is not None:
                symbolic_state_changed = True

        if self.done_if_solved:
            done = self.is_solved()
            reward = 1 if done else 0
            info["is_success"] = done
        else:
            done = symbolic_state_changed
            reward = 0

        obs = self.get_current_observation()
        return obs, reward, done, info

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        self.board.plot_board(ax, board_state, initial_board_state)

    def plot_trajectory(
        self,
        episode,
        ax,
        plot_done=True,
        scatter_kwargs=None,
    ):
        positions = [episode[0]["state"]] + [t["next_state"] for t in episode]
        positions = np.stack(positions)
        actions = np.array([t["action"] for t in episode])
        last_is_done = episode[-1]["done"]

        push_actions = actions[:, -1]
        if self.mixed_action_space:
            activations = push_actions.astype(bool)
        else:
            activations = push_actions > self.PUSH_THRESHOLD
        activations = np.concatenate((np.array([False]), activations))

        plot_2d_trajectory(
            positions,
            activations,
            last_is_done,
            ax,
            plot_done,
            scatter_kwargs,
        )

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
        return plot_skill_episodes(
            self,
            episode_list,
            num_skills,
            subplots_kwargs,
            return_ax,
            ax_arr,
            plot_done,
            scatter_kwargs,
        )

    def render(self, mode="human"):
        if mode != "rgb_array":
            raise NotImplementedError

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plot_board(ax, board_state=self.board_state)
        ax.scatter(*self.cursor, marker="x", s=250, linewidth=5, color="darkblue")

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
