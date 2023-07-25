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
Print number of unique boards per solution depth and split for TileSwap and LightsOut
"""

from tabulate import tabulate

from seads_envs.board_games.board_loader import BoardLoader

# To be run as a script
__all__ = []


def analyze_dataset(hdf_filename, boardsize, style):
    board_loader = BoardLoader(
        hdf_filename,
        boardsize=boardsize,
    )
    solution_depths = ["sol. depth"]
    train_num = ["train"]
    test_num = ["test"]
    all_num = ["all"]
    for sd in range(
        1,
        max(list(board_loader.train_idxs.keys()) + list(board_loader.test_idxs.keys()))
        + 1,
    ):
        solution_depths.append(sd)
        train_num.append(
            len(board_loader.train_idxs[sd]) if sd in board_loader.train_idxs else 0
        )
        test_num.append(
            len(board_loader.test_idxs[sd]) if sd in board_loader.test_idxs else 0
        )
        all_num.append(train_num[-1] + test_num[-1])
    if style == "latex":
        print(" & ".join(map(str, solution_depths)))
        print(" & ".join(map(str, train_num)))
        print(" & ".join(map(str, test_num)))
        print(" & ".join(map(str, all_num)))
    elif style == "table":
        print(tabulate([train_num, test_num, all_num], headers=solution_depths))
    else:
        raise ValueError


def main():
    style = "table"  # or latex

    print("LightsOut")
    analyze_dataset("lightsout_boards_s5.h5py", boardsize=5, style=style)

    print("")
    print("TileSwap")
    analyze_dataset("tileswap_boards_s3.h5py", boardsize=3, style=style)


if __name__ == "__main__":
    main()
