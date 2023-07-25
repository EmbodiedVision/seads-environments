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
LightsOut board
"""

from collections import namedtuple

import numpy as np
from dm_control import composer, mjcf

# --------
# Code block from <PPG>/board_games/_internal/boards.py, modified
_TOUCH_THRESHOLD = 1e-3  # Activation threshold for touch sensors (N).
_SHOW_DEBUG_GRID = False
# --------


LightsOutBoardContact = namedtuple(
    "LightsOutBoardContact", "board_pos norm_pos row col"
)


# Code block from <PPG>/board_games/_internal/boards.py, modified
def _make_lightsoutboard(
    rows, columns, square_halfwidth, sensor_size=0.7, name="checkerboard", stairs=False
):
    """Builds a checkerboard with touch sensors centered on each square."""

    root = mjcf.RootElement(model=name)
    on_bright = root.asset.add(
        "material", name="on_bright", rgba=(230 / 256, 58 / 256, 58 / 256, 1)
    )
    on_dark = root.asset.add(
        "material", name="on_dark", rgba=(154 / 256, 38 / 256, 38 / 256, 1)
    )
    off_bright = root.asset.add(
        "material",
        name="off_bright",
        rgba=(230 / 256, 230 / 256, 230 / 256, 1),
        specular=0.01,
    )
    off_dark = root.asset.add(
        "material",
        name="off_dark",
        rgba=(154 / 256, 154 / 256, 154 / 256, 1),
        specular=0.01,
    )
    # on bright: 230 / 58 / 58 (HSV: 0 / 75 / 90)
    # on dark: 154 / 38 / 38 (HSV: 0 / 75 / 60)
    # off bright: 230 / 230 / 230 (HSV: 0 / 0 / 90)
    # off dark: 154 / 154 / 154 (HSV: 0 / 0 / 60)

    root.default.geom.set_attributes(type="box")

    sensor_mat = root.asset.add("material", name="sensor", rgba=(0, 1, 0, 0.3))
    sensor_height = 0.01
    root.default.site.set_attributes(
        type="box",
        size=(sensor_size * square_halfwidth,) * 2 + (0.5 * sensor_height,),
        material=sensor_mat,
        group=composer.SENSOR_SITES_GROUP,
    )

    xpos = (np.arange(columns) - 0.5 * (columns - 1)) * 2 * square_halfwidth
    ypos = (np.arange(rows) - 0.5 * (rows - 1)) * 2 * square_halfwidth

    board_extent = (
        (xpos[0] - square_halfwidth, ypos[0] - square_halfwidth),
        (xpos[-1] + square_halfwidth, ypos[-1] + square_halfwidth),
    )

    # move board origin from bottomleft to topleft corner
    ypos = np.flip(ypos)

    geoms = []
    touch_sensors = []
    materials = {
        "on_bright": on_bright,
        "on_dark": on_dark,
        "off_bright": off_bright,
        "off_dark": off_dark,
    }
    if stairs:
        height_per_dist = np.linspace(0.01, 0.05, 3)
    else:
        height_per_dist = [
            0.01,
        ] * columns

    i_j_list = []
    if stairs:
        for j in range(columns):
            for i in range(rows):
                i_j_list.append((i, j))
    else:
        for i in range(rows):
            for j in range(columns):
                i_j_list.append((i, j))

    for i, j in i_j_list:
        center_dist = max(abs(j - (columns - 1) / 2), abs(i - (rows - 1) / 2))
        h = height_per_dist[int(center_dist)]
        geom_mat = off_dark if ((i % 2) == (j % 2)) else off_bright
        name = "{}_{}".format(i, j)
        geoms.append(
            root.worldbody.add(
                "geom",
                size=(square_halfwidth, square_halfwidth, h),
                pos=(xpos[j], ypos[i], h),
                name=name,
                material=geom_mat,
            )
        )
        site = root.worldbody.add("site", pos=(xpos[j], ypos[i], 2 * h), name=name)
        touch_sensors.append(root.sensor.add("touch", site=site, name=name))

    return root, geoms, touch_sensors, materials, board_extent


class LightsOutBoard(composer.Entity):
    # The structure of this class is inspired by the `CheckerBoard` class in
    # `<PPG>/board_games/_internal/boards.py`.

    def __init__(self, board_size, stairs, *args, **kwargs):
        self.board_size = board_size
        self._stairs = stairs
        self._contact_from_before_substep = None
        super(LightsOutBoard, self).__init__(*args, **kwargs)

    def _build(self, square_halfwidth=0.05):
        """Builds a `LightsOutBoard` entity.

        Args:
          square_halfwidth: Float, the halfwidth of the squares on the board.
        """
        rows = self.board_size
        columns = self.board_size
        root, geoms, touch_sensors, materials, board_extent = _make_lightsoutboard(
            rows=rows,
            columns=columns,
            square_halfwidth=square_halfwidth,
            stairs=self._stairs,
        )
        self._mjcf_model = root
        self._geoms = np.array(geoms).reshape(rows, columns)
        self._touch_sensors = np.array(touch_sensors).reshape(rows, columns)
        self._materials = materials
        self._square_halfwidth = square_halfwidth
        self.board_extent = board_extent

    def update_board_appearance(self, physics, board_state):
        rows, columns = board_state.shape
        for i in range(rows):
            for j in range(columns):
                prefix = "on" if board_state[i, j] else "off"
                suffix = "dark" if ((i % 2) == (j % 2)) else "bright"
                g = physics.bind(self._geoms[i, j])
                g.matid = {
                    "on_bright": 1,
                    "on_dark": 2,
                    "off_bright": 3,
                    "off_dark": 4,
                }[prefix + "_" + suffix]

    @property
    # Code block from <PPG>/board_games/_internal/boards.py, unmodified
    def mjcf_model(self):
        return self._mjcf_model

    # Code block from <PPG>/board_games/_internal/boards.py, modified
    def before_substep(self, physics, random_state):
        del random_state  # Unused.
        # Cache a copy of the array of active contacts before each substep.
        self._contact_from_before_substep = physics.data.contact.copy()

    # Code block from <PPG>/board_games/_internal/boards.py, modified
    def get_contact_pos(self, physics, row, col):
        geom_id = physics.bind(self._geoms[row, col]).element_id
        # Here we use the array of active contacts from the previous substep, rather
        # than the current values in `physics.data.contact`. This is because we use
        # touch sensors to detect when a square on the board is being pressed, and
        # the pressure readings are based on forces that were calculated at the end
        # of the previous substep. It's possible that `physics.data.contact` no
        # longer contains any active contacts involving the board geoms, even though
        # the touch sensors are telling us that one of the squares on the board is
        # being pressed.
        contact = self._contact_from_before_substep
        involves_geom = (contact.geom1 == geom_id) | (contact.geom2 == geom_id)
        [relevant_contact_ids] = np.where(involves_geom)
        if relevant_contact_ids.size:
            # If there are multiple contacts involving this square of the board, just
            # pick the first one.
            return contact[relevant_contact_ids[0]].pos.copy()
        else:
            print(
                "Touch sensor at ({},{}) doesn't have any active contacts!".format(
                    row, col
                )
            )
            return None

    # Code block from <PPG>/board_games/_internal/boards.py, unmodified
    def get_contact_indices(self, physics):
        pressures = physics.bind(self._touch_sensors.ravel()).sensordata
        # If any of the touch sensors exceed the threshold, return the (row, col)
        # indices of the most strongly activated sensor.
        if np.any(pressures > _TOUCH_THRESHOLD):
            return np.unravel_index(np.argmax(pressures), self._touch_sensors.shape)
        else:
            return None

    def get_contact_info(self, physics):
        indices = self.get_contact_indices(physics)
        if indices is None:
            return None
        row, col = indices
        contact_pos = self.get_contact_pos(physics, row, col)
        if contact_pos is None:
            return None
        board_width = self.board_size * 2 * self._square_halfwidth
        norm_pos = (contact_pos[:2] + board_width / 2) / board_width
        norm_pos[1] = 1 - norm_pos[1]
        contact_info = LightsOutBoardContact(
            board_pos=contact_pos, norm_pos=norm_pos, row=row, col=col
        )
        return contact_info
