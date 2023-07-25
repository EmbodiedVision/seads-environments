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

import unittest

import numpy as np

from seads_envs.hybrid.cursor import LightsOutCursorEnv, TileSwapCursorEnv
from seads_envs.hybrid.jaco import LightsOutJacoEnv, TileSwapJacoEnv
from seads_envs.hybrid.reacher import LightsOutReacherEnv, TileSwapReacherEnv

DEFAULT_ENV_KWARGS = {
    "reset_split": "train",
    "max_solution_depth": 5,
    "random_solution_depth": False,
}


class TestParametrizedEnv(unittest.TestCase):
    def setUp(self) -> None:
        pass

    # (Seed / reset) twice should yield the same initial state
    # same for (Seed / reset / reset)
    def assert_reproducibility_reseed(self, env_class, env_kwargs):
        env = env_class(**env_kwargs)
        for reset_twice in [True, False]:
            for seed in range(10):
                env.seed(seed)
                if reset_twice:
                    env.reset()
                state_1 = env.reset()
                env.seed(seed)
                if reset_twice:
                    env.reset()
                state_2 = env.reset()
                np.testing.assert_equal(state_1, state_2)

    # (seed / reset / reset) should yield initial states which differ from (seed / reset),
    # both in discrete state and continuous state
    def assert_difference_reset(self, env_class, env_kwargs):
        env = env_class(**env_kwargs)
        for seed in range(10):
            env.seed(seed)
            state_1 = env.reset()
            manip_state_1 = env.get_manipulator_state(state_1)
            symb_state_1 = env.get_symbolic_state(state_1)
            state_2 = env.reset()
            manip_state_2 = env.get_manipulator_state(state_2)
            symb_state_2 = env.get_symbolic_state(state_2)
            self.assertFalse(np.array_equal(manip_state_1, manip_state_2))
            self.assertFalse(np.array_equal(symb_state_1, symb_state_2))

    # reset of hybrid envs with some seed should yield same board as reset of discrete envs
    def assert_equality_reset_discrete_env(self, env_class, env_kwargs):
        env = env_class(**env_kwargs)
        for seed in range(10):
            env.seed(seed)
            state_1 = env.reset()
            symb_state_1 = env.get_symbolic_state(state_1)
            env.discrete_env.seed(seed)
            symb_state_2 = env.discrete_env.reset()
            np.testing.assert_equal(symb_state_1, symb_state_2)

    def assert_pass_single_env(self, env_class, env_kwargs):
        for test in [
            self.assert_reproducibility_reseed,
            self.assert_difference_reset,
            self.assert_equality_reset_discrete_env,
        ]:
            with self.subTest(msg=f"{env_class.__name__} - {test.__name__}"):
                test(env_class, env_kwargs)

    def test_lightsout_cursor(self):
        env_class = LightsOutCursorEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)

    def test_tileswap_cursor(self):
        env_class = TileSwapCursorEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)

    def test_lightsout_reacher(self):
        env_class = LightsOutReacherEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)

    def test_tileswap_reacher(self):
        env_class = TileSwapReacherEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)

    def test_lightsout_jaco(self):
        env_class = LightsOutJacoEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)

    def test_tileswap_jaco(self):
        env_class = TileSwapJacoEnv
        self.assert_pass_single_env(env_class, DEFAULT_ENV_KWARGS)
