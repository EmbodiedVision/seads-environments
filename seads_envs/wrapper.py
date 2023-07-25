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

import gym
import numpy as np
from gym import Wrapper


class TimeLimitWrapper(Wrapper):
    """
    Wraps a Gym environment to terminate after a finite number of steps.
    In addition, the observation is augmented to indicate if
    the current observation is intermediate ([0 0]), second-to last ([1 0],
    env. will terminate with next step) or last ([0 1], env. terminated).
    """

    def __init__(self, env, max_steps, ext_timelimit_obs):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_steps = max_steps
        self._steps = 0
        self._ext_timelimit_obs = ext_timelimit_obs

        if ext_timelimit_obs == "full":
            self._extra_dims = 1
        elif type(ext_timelimit_obs) == bool and ext_timelimit_obs == False:
            self._extra_dims = 0
        else:
            raise ValueError

        if self._extra_dims > 0:
            if type(env.observation_space) == gym.spaces.Box:
                observation_space = gym.spaces.Box(
                    low=np.concatenate(
                        (
                            env.observation_space.low,
                            np.zeros(self._extra_dims),
                        )
                    ).astype(np.float32),
                    high=np.concatenate(
                        (
                            env.observation_space.high,
                            np.ones(self._extra_dims),
                        )
                    ).astype(np.float32),
                    dtype=np.float32,
                )
            elif type(env.observation_space) == gym.spaces.Space:
                assert len(env.observation_space.shape) == 1
                new_shape = (env.observation_space.shape[0] + self._extra_dims,)
                observation_space = gym.spaces.Space(
                    new_shape, dtype=env.observation_space.dtype
                )
            else:
                raise ValueError(f"Invalid type {type(env.observation_space)}")
        else:
            observation_space = env.observation_space
        self.observation_space = observation_space

    def _extend_obs(self, base_obs):
        if self._ext_timelimit_obs == "full":
            ext_arr = np.array([(self._max_steps - self._steps) / self._max_steps])
            obs = np.concatenate((base_obs, ext_arr))
        else:
            obs = base_obs
        return obs

    def get_symbolic_state(self, obs):
        # strip off additional information
        if self._extra_dims > 0:
            return self.env.get_symbolic_state(obs[..., : -self._extra_dims])
        else:
            return self.env.get_symbolic_state(obs)

    def get_binary_symbolic_state(self, obs):
        # strip off additional information
        if self._extra_dims > 0:
            return self.env.get_binary_symbolic_state(obs[..., : -self._extra_dims])
        else:
            return self.env.get_binary_symbolic_state(obs)

    def get_current_observation(self):
        return self._extend_obs(self.env.get_current_observation())

    def reset(self):
        self._steps = 0
        base_obs = self.env.reset()
        ext_obs = self._extend_obs(base_obs)
        return ext_obs

    def reset_step_counter(self):
        self._steps = 0

    def step(self, action):
        base_obs, reward, done, info = self.env.step(action)
        self._steps += 1
        if self._steps > self._max_steps:
            raise RuntimeError("env should have terminated")
        ext_obs = self._extend_obs(base_obs)
        if self._steps == self._max_steps:
            done = True
            info["timeout"] = True
        return ext_obs, reward, done, info
