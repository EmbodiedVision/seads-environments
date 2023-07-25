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

from abc import ABC

import gym
import numpy as np


class WrappedDiscreteEnv(gym.Env, ABC):
    """
    Wrap discrete env with a manipulator
    """

    def __init__(
        self, discrete_env, manipulator_action_space, manipulator_observation_space
    ):
        self._discrete_env = discrete_env
        self._binary_state_dim = np.prod(np.array(discrete_env.binary_symbolic_shape))
        self._manip_observation_space = manipulator_observation_space
        self._action_space = manipulator_action_space
        self._observation_space = self.wrap_observation_space(
            manipulator_observation_space
        )
        self.rng = None

    @property
    def board_state(self):
        return self._discrete_env.board_state

    @property
    def board_size(self):
        return self._discrete_env.board_size

    @property
    def discrete_env(self):
        return self._discrete_env

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def wrap_observation_space(self, base_space):
        if type(base_space) == gym.spaces.Box:
            zeros = np.zeros(self._binary_state_dim)
            ones = 1 + zeros
            new_low = np.concatenate((base_space.low, zeros))
            new_high = np.concatenate((base_space.high, ones))
            observation_space = gym.spaces.Box(
                low=new_low.astype(np.float32),
                high=new_high.astype(np.float32),
                dtype=np.float32,
            )
        elif type(base_space) == gym.spaces.Space:
            assert len(base_space.shape) == 1
            new_shape = (base_space.shape[0] + self._binary_state_dim,)
            observation_space = gym.spaces.Space(new_shape, dtype=base_space.dtype)
        else:
            raise ValueError
        return observation_space

    @property
    def n_groundtruth_skills(self):
        return self._discrete_env.action_space.n

    @property
    def symbolic_shape(self):
        return self._discrete_env.symbolic_shape

    @property
    def binary_symbolic_shape(self):
        return self._discrete_env.binary_symbolic_shape

    @property
    def symbolic_state(self):
        return self._discrete_env.symbolic_state

    @property
    def binary_symbolic_state(self):
        return self._discrete_env.binary_symbolic_state

    @property
    def symbolic_target_state(self):
        return self._discrete_env.symbolic_target_state

    @property
    def binary_symbolic_target_state(self):
        return self._discrete_env.binary_symbolic_target_state

    def is_solved(self):
        return self._discrete_env.is_solved()

    @property
    def max_solution_depth(self):
        return self._discrete_env.max_solution_depth

    @max_solution_depth.setter
    def max_solution_depth(self, value):
        self._discrete_env.max_solution_depth = value

    def _get_manipulator_observation(self):
        raise NotImplementedError

    def get_current_observation(self, manipulator_obs=None):
        binary_symbolic_state = self.binary_symbolic_state.reshape(-1)
        if manipulator_obs is None:
            manipulator_obs = self._get_manipulator_observation()
        order = (manipulator_obs, binary_symbolic_state.astype(float))
        return np.copy(np.concatenate(order))

    def get_manipulator_state(self, obs):
        manipulator_state_dim = self._manip_observation_space.shape[0]
        manipulator_state = obs[..., :manipulator_state_dim]
        return manipulator_state

    def get_binary_symbolic_state(self, obs):
        assert obs.shape[-1:] == self.observation_space.shape
        batchsize = obs.shape[:-1]
        state_dim = self._binary_state_dim
        binary_state_flat = obs[..., -state_dim:]
        binary_state = binary_state_flat.reshape(
            *batchsize, *self._discrete_env.binary_symbolic_shape
        )
        if isinstance(binary_state, np.ndarray):
            binary_state = binary_state.astype(bool)
        else:
            binary_state = binary_state.bool()
        return binary_state

    def get_symbolic_state(self, obs):
        assert obs.shape[-1:] == self.observation_space.shape
        binary_symbolic_state = self.get_binary_symbolic_state(obs)
        return self._discrete_env.binary_to_state(binary_symbolic_state)

    def set_binary_symbolic_state(self, binary_symbolic_state):
        assert binary_symbolic_state.shape == self.binary_symbolic_shape
        assert binary_symbolic_state.dtype == bool
        self._discrete_env.set_binary_symbolic_state(binary_symbolic_state)

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self._discrete_env.seed(seed)
        return [seed]

    @property
    def all_rngs(self):
        return {"wrapping_rng": self.rng, "wrapped_rng": self._discrete_env.rng}

    @all_rngs.setter
    def all_rngs(self, new_rngs):
        self.rng = new_rngs["wrapping_rng"]
        self._discrete_env.rng = new_rngs["wrapped_rng"]

    @property
    def reset_split(self):
        return self._discrete_env.reset_split

    @reset_split.setter
    def reset_split(self, new_reset_split):
        self._discrete_env.reset_split = new_reset_split

    def step(self, action):
        raise NotImplementedError

    def plot_board(self, ax, board_state=None, initial_board_state=None):
        raise NotImplementedError

    def plot_trajectory(
        self,
        episode,
        ax,
        plot_done=True,
        scatter_kwargs=None,
    ):
        raise NotImplementedError

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
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError
