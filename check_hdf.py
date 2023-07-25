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

HDF_FILES_GLOB = "seads_envs/board_games/*.h5py"


def is_hdf_file(filename):
    magic_numbers = bytes([0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
    with open(filename, "rb") as fd:
        file_head = fd.read(len(magic_numbers))
    return file_head.startswith(magic_numbers)


def check_hdf_files():
    import glob

    ERRMSG = (
        "\n"
        "\n"
        "File at '{filename}' is not a valid HDF file! \n"
        "Please clone the 'seads-environments' repository again "
        "with git-lfs installed (see README.md of 'seads-environments')."
        "\n"
        "\n"
    )

    fail = False
    for filename in glob.glob(HDF_FILES_GLOB):
        if not is_hdf_file(filename):
            msg = ERRMSG.format(filename=filename)
            print("\033[93m " + msg + " \033[0m")
            fail = True

    if not fail:
        print("All HDF files are valid, please continue with installation.")


if __name__ == "__main__":
    check_hdf_files()
