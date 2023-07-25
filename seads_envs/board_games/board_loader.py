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
Load pre-computed boards with specified solution depths from HDF5 files
"""
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "BoardLoader",
    "BoardLoaderError",
]


class BoardLoaderError(Exception):
    pass


class BoardLoader:
    """
    Load precomputed board configurations for a particular solution depth.
    """

    def __init__(self, filename, boardsize):
        h5_file = Path(__file__).resolve().parent.joinpath(filename)
        self.train_idxs = {}
        self.test_idxs = {}
        self.n_boards_per_sd = {}
        self.f = h5py.File(h5_file, "r")
        assert self.f.attrs["boardsize"] == boardsize
        max_query_solution_depth = self.f.attrs["max_solution_depth"]
        for solution_depth in range(1, max_query_solution_depth + 1):
            if f"splits_sd{solution_depth}" not in self.f:
                continue
            splits = np.array(self.f[f"splits_sd{solution_depth}"])
            self.n_boards_per_sd[solution_depth] = len(splits)
            # 'splits' are saved as bytearrays
            self.train_idxs[solution_depth] = np.where(splits == b"train")[0]
            self.test_idxs[solution_depth] = np.where(splits == b"test")[0]

    def load_packed_board(self, solution_depth, split, rng):
        if solution_depth not in self.n_boards_per_sd.keys():
            raise BoardLoaderError(f"Unavailable solution depth {solution_depth}")
        if split == "all":
            n_boards = self.n_boards_per_sd[solution_depth]
            idx = rng.randint(0, n_boards)
        elif split == "train":
            idx = rng.choice(self.train_idxs[solution_depth])
        elif split == "test":
            idx = rng.choice(self.test_idxs[solution_depth])
        else:
            raise ValueError
        board = self.f[f"boards_sd{solution_depth}"][idx]
        actual_split = self.f[f"splits_sd{solution_depth}"][idx].decode()
        if split in ["train", "test"]:
            if not split == actual_split:
                raise RuntimeError("split mismatch")
        return board
