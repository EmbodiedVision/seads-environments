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
Board representations for LightsOut and TileSwap discrete environments
"""

import math
import zlib

import numpy as np

__all__ = [
    "classify_board_split",
    "LightsOutBoardRepr",
    "TileSwapBoardRepr",
]


def compute_board_state_str(canonical_board_state):
    """ Compute (comma-separated) string representation of board """
    if canonical_board_state.dtype == bool:
        canonical_board_state = canonical_board_state.astype(int)
    board_state_str = ",".join([str(i) for i in canonical_board_state.ravel()])
    return board_state_str


def classify_board_split(canonical_board_state, return_hash=False):
    """
    Classify a board state into "train", "test"
    based on the integer division remainder of its hash.

    Parameters
    ----------
    canonical_board_state: `np.ndarray`, shape [N, N]
        2-dimensional board
    return_hash: bool
        Return hash code of board

    Returns
    -------
    split: str
        Split, in ["train", "test"]
    hash_: int, optional
        Hash code
    """
    assert canonical_board_state.ndim == 2
    board_state_str = compute_board_state_str(canonical_board_state)
    hash_ = zlib.crc32(board_state_str.encode())
    # Classify board into split based on integer division remainder
    remainder = np.mod(hash_, 3)
    if remainder == 0:
        split = "train"
    else:
        split = "test"
    if return_hash:
        return split, hash_
    else:
        return split


class LightsOutBoardRepr:
    def __init__(self, boardsize):
        self.boardsize = boardsize

    @property
    def initial_binary_state(self):
        return np.zeros((self.boardsize, self.boardsize)).astype(bool)

    def binary_to_packed(self, binary_boards):
        assert binary_boards.dtype == bool
        assert binary_boards.shape[-2:] == (self.boardsize, self.boardsize)
        binary_flat = binary_boards.reshape(
            *binary_boards.shape[:-2], self.boardsize ** 2
        )
        packed = np.packbits(binary_flat, axis=-1)
        return packed

    def packed_to_binary(self, packed_boards):
        assert packed_boards.dtype == np.uint8
        assert packed_boards.shape[-1] == math.ceil(self.boardsize ** 2 / 8)
        unpacked = np.unpackbits(
            packed_boards, axis=-1, count=self.boardsize ** 2
        ).astype(bool)
        binary_boards = unpacked.reshape(
            (*unpacked.shape[:-1], self.boardsize, self.boardsize)
        )
        return binary_boards

    def binary_to_canonical(self, binary_boards):
        assert binary_boards.dtype == bool
        assert tuple(binary_boards.shape[-2:]) == (self.boardsize, self.boardsize)
        return binary_boards

    def packed_to_canonical(self, packed_boards):
        binary_boards = self.packed_to_binary(packed_boards)
        canonical_boards = self.binary_to_canonical(binary_boards)
        return canonical_boards

    def packed_to_fingerprint(self, packed_board):
        assert packed_board.ndim == 1
        return packed_board.tobytes().hex()


class TileSwapBoardRepr:
    def __init__(self, boardsize):
        if boardsize != 3:
            raise NotImplementedError
        self.boardsize = boardsize

    @property
    def initial_binary_state(self):
        return np.eye(self.boardsize ** 2).astype(bool)

    def binary_to_packed(self, binary_boards):
        assert binary_boards.dtype == bool
        assert binary_boards.shape[-2:] == (self.boardsize ** 2, self.boardsize ** 2)
        binary_flat = binary_boards.reshape(
            *binary_boards.shape[:-2], self.boardsize ** 4
        )
        packed = np.packbits(binary_flat, axis=-1)
        return packed

    def packed_to_binary(self, packed_boards):
        assert packed_boards.dtype == np.uint8
        assert packed_boards.shape[-1] == math.ceil(self.boardsize ** 4 / 8)
        unpacked = np.unpackbits(
            packed_boards, axis=-1, count=self.boardsize ** 4
        ).astype(bool)
        binary_boards = unpacked.reshape(
            (*unpacked.shape[:-1], self.boardsize ** 2, self.boardsize ** 2)
        )
        return binary_boards

    def binary_to_canonical(self, binary_boards):
        assert binary_boards.dtype == bool
        assert binary_boards.shape[-2:] == (self.boardsize ** 2, self.boardsize ** 2)
        argmax = np.argmax(binary_boards, axis=-1)
        batch_shape = binary_boards.shape[:-2]
        canonical_board = argmax.reshape(batch_shape + (self.boardsize, self.boardsize))
        return canonical_board

    def packed_to_canonical(self, packed_boards):
        binary_boards = self.packed_to_binary(packed_boards)
        canonical_boards = self.binary_to_canonical(binary_boards)
        return canonical_boards

    def packed_to_fingerprint(self, packed_board):
        assert packed_board.ndim == 1
        return packed_board.tobytes().hex()
