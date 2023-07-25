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

from os import path

import numpy as np
from gym import spaces
from mujoco_py import MjSim, MjSimState, load_model_from_path
from mujoco_py.modder import TextureModder

from seads_envs.hybrid.plotting import plot_2d_trajectory, plot_skill_episodes
from seads_envs.hybrid.wrapped_discrete import WrappedDiscreteEnv

__all__ = ["ReacherEnv"]


class ReacherEnv(WrappedDiscreteEnv):
    def __init__(
        self,
        discrete_env,
        touch_board,
        mixed_action_space,
        done_if_solved,
        action_repeat,
        render_width,
        render_height,
    ):
        if mixed_action_space:
            continuous_space = spaces.Box(
                low=np.array([-1, -1]).astype(np.float32),
                high=np.array([1, 1]).astype(np.float32),
                dtype=np.float32,
            )
            discrete_space = spaces.Discrete(n=2)
            action_space = spaces.Tuple(spaces=(continuous_space, discrete_space))
        else:
            action_space = spaces.Box(
                low=np.array([-1, -1, -1]).astype(np.float32),
                high=np.array([1, 1, 1]).astype(np.float32),
                dtype=np.float32,
            )
            self.PUSH_THRESHOLD = -1 + 0.8 * 2

        super().__init__(
            discrete_env,
            manipulator_action_space=action_space,
            manipulator_observation_space=spaces.Space((4,), dtype=float),
        )
        self.board = touch_board
        self.action_repeat = action_repeat
        self._render_width = render_width
        self._render_height = render_height

        self._sim = MjSim(
            load_model_from_path(
                path.join(path.dirname(__file__), "xmls", "reacher.xml")
            )
        )
        self.sim = self._sim
        self._modder = TextureModder(self._sim)

        self.mixed_action_space = mixed_action_space
        self.done_if_solved = done_if_solved

    def reset(self):
        self.discrete_env.reset()
        position = self.rng.uniform(0, 2 * np.pi, 2)
        velocity = np.zeros(2)  # self.rng.uniform(-0.3, 0.3, 2)
        new_state = MjSimState(
            time=0, qpos=position, qvel=velocity, act=None, udd_state={}
        )
        self._sim.reset()
        self._sim.set_state(new_state)
        self._sim.forward()
        return self.get_current_observation()

    def _get_manipulator_observation(self):
        sim_state = self._sim.get_state()
        return np.concatenate([sim_state.qpos, sim_state.qvel]).copy()

    def _process_action(self, action):
        if self.mixed_action_space:
            reacher_ctrl, push = action
        else:
            reacher_ctrl = action[:-1]
            push_real = action[-1]
            push = push_real > self.PUSH_THRESHOLD
        return reacher_ctrl, push

    def _register_push(self, pos):
        raise NotImplementedError

    def _get_finger_pos(self):
        raw_pos = np.copy(self._sim.data.get_geom_xpos("finger")[:2])
        norm_pos = (raw_pos[:2] + 0.15) / 0.3
        return norm_pos

    def step(self, action):
        reacher_ctrl, push = self._process_action(action)

        info = {"pos_finger": self._get_finger_pos()}

        for _ in range(self.action_repeat):
            self._sim.data.ctrl[:] = reacher_ctrl
            self._sim.step()

        self._sim.forward()

        next_pos_finger = self._get_finger_pos()
        info["next_pos_finger"] = next_pos_finger

        cursor = next_pos_finger

        groundtruth_skill = None
        if push and (cursor <= 1).all() and (cursor >= 0).all():
            groundtruth_skill = self._register_push(cursor)
            info["groundtruth_skill"] = groundtruth_skill

        if self.done_if_solved:
            done = self.is_solved()
            reward = 1 if done else 0
            info["is_success"] = done
        else:
            done = groundtruth_skill is not None
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
        positions = [episode[0]["info"]["pos_finger"]] + [
            t["info"]["next_pos_finger"] for t in episode
        ]
        positions = np.stack(positions)
        push_actions = np.array([t["action"][-1] for t in episode])

        if self.mixed_action_space:
            activations = push_actions.astype(bool)
        else:
            activations = push_actions > self.PUSH_THRESHOLD
        activations = np.concatenate((np.array([False]), activations))
        last_is_done = episode[-1]["done"]
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
        self._set_texture()
        rgb_array = self._sim.render(
            self._render_width, self._render_height, camera_name="fixed"
        )
        return rgb_array

    def _set_texture(self):
        raise NotImplementedError
