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
Generate initial configurations of boards with particular solution depth through reverse BFS
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from seads_envs.board_games.bfs_dynamics import (
    LightsOutBfsDynamics,
    TileSwapBfsDynamics,
)
from seads_envs.board_games.board_repr import LightsOutBoardRepr, TileSwapBoardRepr

# To be run as a script
__all__ = []

this_dir = Path(__file__).parent.resolve()


def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i : i + chunk_size]


def reverse_bfs(board_repr, bfs_dynamics, max_solution_depth):
    initial_binary_state = board_repr.initial_binary_state
    assert initial_binary_state.dtype == bool
    n_steps = max_solution_depth
    batchsize = 50_000

    initial_packed_state = board_repr.binary_to_packed(initial_binary_state)

    explored_per_sd = {k: {} for k in range(1, max_solution_depth + 1)}
    explored_per_sd[0] = {
        board_repr.packed_to_fingerprint(initial_packed_state): initial_packed_state
    }

    def _fingerprint_seen(fingerprint, max_explored_sd):
        found = False
        for sd_ in reversed(range(0, max_explored_sd + 1)):
            if fingerprint in explored_per_sd[sd_]:
                found = True
                break
        return found

    fringe = [initial_packed_state]

    for step in range(n_steps):
        if len(fringe) == 0:
            break

        solution_depth = step + 1

        new_fringe = []

        for batch_states_packed in tqdm(list(split(fringe, batchsize))):
            batch_states_packed = np.stack(batch_states_packed)
            batch_states_binary = board_repr.packed_to_binary(batch_states_packed)
            next_states_binary = bfs_dynamics.expand_binary_boards(batch_states_binary)
            next_states_packed = board_repr.binary_to_packed(next_states_binary)

            for next_state in next_states_packed:
                next_fingerprint = board_repr.packed_to_fingerprint(next_state)
                if not _fingerprint_seen(next_fingerprint, solution_depth):
                    explored_per_sd[solution_depth][next_fingerprint] = next_state
                    new_fringe.append(next_state)

        fringe = new_fringe
        print(f"Explored all boards with solution depth {step+1}: {len(new_fringe)}")

    return explored_per_sd


def store_boards(board_repr, bfs_dynamics, max_solution_depth, filename):
    from seads_envs.board_games.board_repr import classify_board_split

    if filename.exists():
        print(f"File {filename} already exists, here is some info about the boards:")
        print_dataset_info(filename)
        return

    # Save boards to HDF5, with datasets per solution depth
    with h5py.File(filename, "w") as f:
        f.attrs["boardsize"] = board_repr.boardsize
        f.attrs["max_solution_depth"] = max_solution_depth

        all_boards_per_sd = reverse_bfs(board_repr, bfs_dynamics, max_solution_depth)
        for solution_depth in tqdm(list(all_boards_per_sd.keys())):
            if len(all_boards_per_sd[solution_depth]) == 0:
                continue
            fingerprints, boards = zip(*list(all_boards_per_sd[solution_depth].items()))
            boards = np.stack(boards)
            fingerprints = np.array(fingerprints, dtype="S")
            canonical_boards = board_repr.packed_to_canonical(boards)
            splits = Parallel(n_jobs=16)(
                delayed(classify_board_split)(b) for b in canonical_boards
            )
            splits = np.array(splits, dtype="S")

            f.create_dataset(f"boards_sd{solution_depth}", data=boards)
            f.create_dataset(f"splits_sd{solution_depth}", data=splits)
            f.create_dataset(f"fingerprints_sd{solution_depth}", data=fingerprints)

        f.flush()


def print_dataset_info(filename):
    with h5py.File(filename, "r") as f:
        # Extract available solution depths, from dataset names (e.g., boards_sd3)
        available_sd = sorted(
            [int(k.split("_")[1][2:]) for k in f.keys() if k.startswith("boards")]
        )
        all_boards = []
        for solution_depth in available_sd:
            splits = f[f"splits_sd{solution_depth}"][:]
            fingerprints = f[f"fingerprints_sd{solution_depth}"][:]
            all_boards.extend(
                zip(
                    [
                        solution_depth,
                    ]
                    * len(splits),
                    splits,
                    fingerprints,
                )
            )
        df = pd.DataFrame(
            all_boards, columns=["solution_depth", "split", "fingerprint"]
        )
        print(
            df.groupby(["solution_depth", "split"])["fingerprint"]
            .count()
            .reset_index()
            .pivot(index="solution_depth", columns="split")
        )


def main():
    lightsout_boardsize = 5
    max_solution_depth = 25
    board_repr = LightsOutBoardRepr(lightsout_boardsize)
    bfs_dyn = LightsOutBfsDynamics()
    store_boards(
        board_repr,
        bfs_dyn,
        max_solution_depth,
        this_dir.joinpath(f"lightsout_boards_s{lightsout_boardsize}.h5py"),
    )

    tileswap_boardsize = 3
    max_solution_depth = 25  # not reached
    board_repr = TileSwapBoardRepr(tileswap_boardsize)
    bfs_dyn = TileSwapBfsDynamics()
    store_boards(
        board_repr,
        bfs_dyn,
        max_solution_depth,
        this_dir.joinpath(f"tileswap_boards_s{tileswap_boardsize}.h5py"),
    )


if __name__ == "__main__":
    main()
