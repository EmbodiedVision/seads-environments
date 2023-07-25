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
TileSwap board
"""


from collections import namedtuple
from pathlib import Path

import numpy as np
from dm_control import composer, mjcf
from dm_control.utils import io as resources

from seads_envs import ASSETS_DIR
from seads_envs.hybrid.touch_board import TileSwapTouchBoard

# --------
# Code block from <PPG>/board_games/_internal/boards.py, unmodified
_TOUCH_THRESHOLD = 1e-3  # Activation threshold for touch sensors (N).
# --------

TileSwapBoardContact = namedtuple(
    "TileSwapBoardContact", "board_pos norm_pos swap_indices"
)


# Code block from <PPG>/board_games/_internal/boards.py, modified
def _make_tileswapboard(board_size, square_halfwidth, height=0.01, name="checkerboard"):
    root = mjcf.RootElement(model=name)

    sensor_mat = root.asset.add("material", name="sensor", rgba=(0, 1, 0, 1))
    root.default.geom.set_attributes(
        type="box", size=(square_halfwidth, square_halfwidth, height)
    )
    root.default.site.set_attributes(
        type="box",
        size=(square_halfwidth * board_size,) * 2 + (0.5 * height,),
        material=sensor_mat,
        group=composer.SENSOR_SITES_GROUP,
    )

    xpos = (np.arange(board_size) - 0.5 * (board_size - 1)) * 2 * square_halfwidth
    ypos = (np.arange(board_size) - 0.5 * (board_size - 1)) * 2 * square_halfwidth

    board_extent = (
        (xpos[0] - square_halfwidth, ypos[0] - square_halfwidth),
        (xpos[-1] + square_halfwidth, ypos[-1] + square_halfwidth),
    )

    # move board origin from bottomleft to topleft corner
    ypos = np.flip(ypos)

    geoms = []

    for i in range(board_size):
        for j in range(board_size):
            texture_path = ASSETS_DIR.joinpath("tileswap", f"{i*board_size + j}.png")
            contents = resources.GetResource(texture_path)
            root.asset.add(
                "texture",
                name=f"{i*board_size + j}",
                type="2d",
                file=mjcf.Asset(contents, ".png"),
            )
            geom_mat = root.asset.add(
                "material",
                name=f"{i*board_size + j}",
                texture=f"{i*board_size + j}",
                texrepeat=[0.97, 0.97],
            )
            name = "{}_{}".format(i, j)
            geoms.append(
                root.worldbody.add(
                    "geom", pos=(xpos[j], ypos[i], height), name=name, material=geom_mat
                )
            )

    # single touch sensor
    site = root.worldbody.add("site", pos=(0, 0, 2 * height), name="touch_site")
    touch_sensor = root.sensor.add("touch", site=site, name="touch_sensor")

    return root, geoms, touch_sensor, board_extent


class TileSwapBoard(composer.Entity):
    # The structure of this class is inspired by the `CheckerBoard` class in
    # `<PPG>/board_games/_internal/boards.py`.

    def __init__(self, board_size, *args, **kwargs):
        self.board_size = board_size
        self._contact_from_before_substep = None
        super(TileSwapBoard, self).__init__(*args, **kwargs)

    # Code block from <PPG>/board_games/_internal/boards.py, modified
    def _build(self, square_halfwidth=0.05):
        """Builds a `TileSwapBoard` entity.

        Args:
          square_halfwidth: Float, the halfwidth of the squares on the board.
        """
        root, geoms, touch_sensor, board_extent = _make_tileswapboard(
            self.board_size, square_halfwidth=square_halfwidth
        )
        self._mjcf_model = root
        self._geoms = np.array(geoms).reshape(self.board_size, self.board_size)
        self._touch_sensor = touch_sensor
        self._board = TileSwapTouchBoard(self.board_size)
        self._square_halfwidth = square_halfwidth
        self.board_extent = board_extent

    def update_board_appearance(self, physics, board_state):
        rows, columns = board_state.shape
        for i in range(rows):
            for j in range(columns):
                g = physics.bind(self._geoms[i, j])
                g.matid = board_state[i, j] + 2

    @property
    # Code block from <PPG>/board_games/_internal/boards.py, unmodified
    def mjcf_model(self):
        return self._mjcf_model

    # Code block from <PPG>/board_games/_internal/boards.py, modified
    def before_substep(self, physics, random_state):
        del random_state  # Unused.
        # Cache a copy of the array of active contacts before each substep.
        self._contact_from_before_substep = physics.data.contact.copy()

    def get_contact_info_from_pos(self, contact_pos):
        # Normalize contact_pos to [0, 1] extent with (0, 0) on topleft of board
        board_width = self.board_size * 2 * self._square_halfwidth
        norm_pos = (contact_pos[:2] + board_width / 2) / board_width
        norm_pos[1] = 1 - norm_pos[1]
        if np.min(norm_pos) >= 0 and np.max(norm_pos) <= 1:
            # Get pair of affected fields through pivot points
            swap_indices = self._board.get_swap_pair_from_pos(norm_pos)
            if swap_indices is None:
                # Touched outside of valid swap region
                return None
            else:
                return TileSwapBoardContact(
                    board_pos=contact_pos[:2],
                    norm_pos=norm_pos,
                    swap_indices=swap_indices,
                )
        else:
            return None

    def get_contact_info(self, physics):
        pressure = physics.bind(self._touch_sensor).sensordata
        if pressure < _TOUCH_THRESHOLD:
            return None

        contact = self._contact_from_before_substep
        # Check all squares for contact
        for geom in self._geoms.ravel():
            geom_id = physics.bind(geom).element_id
            involves_geom = (contact.geom1 == geom_id) | (contact.geom2 == geom_id)
            [relevant_contact_ids] = np.where(involves_geom)
            if relevant_contact_ids.size:
                # If there are multiple contacts involving the touch sensor just pick the first one.
                contact_pos = contact[relevant_contact_ids[0]].pos.copy()
                contact_info = self.get_contact_info_from_pos(contact_pos)
                return contact_info

        return None
