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
General class for manipulator-embedded board games and Jaco manipulator
"""

import collections
import functools

import numpy as np
from absl import logging
from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.variation import distributions, rotations
from dm_control.entities.manipulators import base, kinova

from seads_envs.third_party.physics_planning_games.board_games.internal import (
    arenas,
    observations,
)
from seads_envs.utils import PatchedFunctionContext

# --------
# Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
_ARM_Y_OFFSET = 0.4
_TCP_LOWER_BOUNDS = (-0.1, -0.1, 0.2)
_TCP_UPPER_BOUNDS = (0.1, 0.1, 0.4)
# --------

SingleMarkerAction = collections.namedtuple("SingleMarkerAction", ["row", "col"])


class IgnoreInverseKinematicsWarnings(PatchedFunctionContext):
    """
    Occasionally, the randomly sampled tool center position is not feasible,
    causing warnings like

    WARNING:absl:Failed to converge after 99 steps: err_norm=0.0107458

    This is not an issue, as in this case, a new random initialization is
    sampled and checked for feasibility.
    See https://github.com/deepmind/dm_control/issues/151.

    This class provides a context manager to suppress this particular warning.
    """

    def __init__(self):
        super().__init__(
            module=logging, fcn_name="warning", fcn_replacement=self._patched_warning
        )

    @staticmethod
    def _patched_warning(msg, *args, **kwargs):
        if msg.startswith("Failed to converge"):
            pass
        else:
            logging.warning(msg, *args, **kwargs)


# --------
# Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
def _uniform_downward_rotation():
    angle = distributions.Uniform(-np.pi, np.pi, single_sample=True)
    quaternion = rotations.QuaternionFromAxisAngle(axis=(0.0, 0.0, 1.0), angle=angle)
    return functools.partial(
        rotations.QuaternionPreMultiply(quaternion), initial_value=base.DOWN_QUATERNION
    )


# --------


class JacoManipulator:
    def __init__(self, observation_settings):

        # --------
        # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
        arm = kinova.JacoArm(
            observable_options=observations.make_options(
                observation_settings, observations.JACO_ARM_OBSERVABLES
            )
        )
        hand = kinova.JacoHand(
            observable_options=observations.make_options(
                observation_settings, observations.JACO_HAND_OBSERVABLES
            )
        )
        # --------

        arm.attach(hand)
        self._arm = arm
        self._hand = hand

        # --------
        # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified

        # Geoms belonging to the arm and hand are placed in a custom group in order
        # to disable their visibility to the top-down camera. NB: we assume that
        # there are no other geoms in ROBOT_GEOM_GROUP that don't belong to the
        # robot (this is usually the case since the default geom group is 0). If
        # there are then these will also be invisible to the top-down camera.
        for robot_geom in arm.mjcf_model.find_all("geom"):
            robot_geom.group = arenas.ROBOT_GEOM_GROUP
        # --------

        # --------
        # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            hand=hand,
            arm=arm,
            position=distributions.Uniform(_TCP_LOWER_BOUNDS, _TCP_UPPER_BOUNDS),
            quaternion=_uniform_downward_rotation(),
        )
        # --------

    def reset(self, physics, random_state):
        with IgnoreInverseKinematicsWarnings():
            self._tcp_initializer(physics, random_state)

    def attach_to_arena(self, arena):
        arena.attach_offset(self._arm, offset=(0, _ARM_Y_OFFSET, 0))


class ManipulatorBoardTask(composer.Task):
    """
    Base class for board games with manipulator.
    """

    # The structure of this class follows the `JacoArmBoardGame` class in
    # `<PPG>/board_games/jaco_arm_board_game.py`.

    def __init__(
        self,
        manipulator,
        board,
        game_logic,
        observation_settings,
        control_timestep,
        reset_arm_after_move,
        rendering_enabled,
    ):
        # --------
        # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
        arena = arenas.Standard(
            observable_options=observations.make_options(
                observation_settings, observations.ARENA_OBSERVABLES
            )
        )
        arena.attach(board)
        # --------

        manipulator.attach_to_arena(arena)

        self._game_logic = game_logic
        self._arena = arena
        self._board = board
        self._manipulator = manipulator
        self._control_timestep = control_timestep
        self._reset_arm_after_move = reset_arm_after_move
        self._rendering_enabled = rendering_enabled
        self._task_observables = {}

        # Define for later use
        self._made_move_this_step = False
        self._contact_info = None
        self._groundtruth_skill = None

    @property
    # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
    def root_entity(self):
        return self._arena

    @property
    # Code block from <PPG>/board_games/jaco_arm_board_game.py, unmodified
    def task_observables(self):
        return self._task_observables

    # Code block from <PPG>/board_games/jaco_arm_board_game.py, modified
    def get_reward(self, physics):
        del physics  # Unused.
        return 0

    # Code block from <PPG>/board_games/jaco_arm_board_game.py, modified
    def should_terminate_episode(self, physics):
        return False

    @property
    def control_timestep(self):
        return self._control_timestep

    def maybe_update_board_appearance(self, physics):
        if self._rendering_enabled:
            self._board.update_board_appearance(
                physics, self._game_logic.get_board_state()
            )

    # Code block from <PPG>/board_games/jaco_arm_board_game.py, modified
    def initialize_episode(self, physics, random_state):
        super(ManipulatorBoardTask, self).initialize_episode(physics, random_state)
        self._manipulator.reset(physics, random_state)
        self._game_logic.reset()
        self.maybe_update_board_appearance(physics)

    # Code block from <PPG>/board_games/jaco_arm_board_game.py, modified
    def before_step(self, physics, action, random_state):
        super(ManipulatorBoardTask, self).before_step(physics, action, random_state)
        self._made_move_this_step = False
        self._contact_info = None

    # Code block from <PPG>/board_games/tictactoe.py, modified
    def after_substep(self, physics, random_state):
        super(ManipulatorBoardTask, self).after_substep(physics, random_state)
        self.maybe_update_board_appearance(physics)
        if not self._made_move_this_step:
            contact_info = self._board.get_contact_info(physics)
            if not contact_info:
                return
            self._contact_info = contact_info
            valid_move, groundtruth_skill = self._game_logic.apply(
                contact_info=contact_info,
            )
            if valid_move:
                self._made_move_this_step = True
                self._groundtruth_skill = groundtruth_skill
                self.maybe_update_board_appearance(physics)
                if not self._game_logic.is_game_over:
                    if self._reset_arm_after_move:
                        self._manipulator.reset(physics, random_state)
            else:
                self._groundtruth_skill = None
