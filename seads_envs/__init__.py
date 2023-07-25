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

import numpy as np

# Override deprecated np.bool, np.float
np.bool = bool
np.float = np.float64

import re
from pathlib import Path

from .wrapper import TimeLimitWrapper

_this_file = Path(__file__).resolve()
ASSETS_DIR = _this_file.parent.joinpath("assets")


def load_env(
    env_name,
    reset_split,
    env_kwargs,
    time_limit=True,
    ext_timelimit_obs="full",
    prototype=False,
):
    """
    Instantiate environment

    Parameters
    ----------
    env_name: str
        Environment name, e.g. LightsOutCursorBs5
    reset_split: str
        Reset split, can be 'train' or 'test'
    env_kwargs: dict
        Dictionary of additional keyword arguments for environment instantiation
    time_limit: bool
        If True, put a time limit on environment execution (defined by 'max_steps' in 'env_kwargs').
    ext_timelimit_obs: str or bool
        Can be 'False' (bool) or 'full' (str). In the latter case,
        the environment observation is extended by a normalized remaining-time counter.
    prototype: bool
        Required for real-world environments
    """
    env_kwargs = dict(**env_kwargs)  # sacred dicts are read-only
    if "Real" in env_name:
        env_kwargs["prototype"] = prototype

    if time_limit:
        # pop 'max_steps' from env_kwargs here, as this is not a kwarg of the environments' __init__
        max_steps = env_kwargs.pop("max_steps")

    # env_name format: LightsOutCursorEnvBs5
    env_name_match = re.fullmatch(r"(.+Env)Bs(\d+)", env_name)
    env_classname = env_name_match.group(1)
    boardsize = int(env_name_match.group(2))

    if env_classname.startswith("LightsOut"):
        if boardsize != 5:
            raise NotImplementedError
    elif env_classname.startswith("TileSwap"):
        if boardsize != 3:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if "Cursor" in env_classname:
        import seads_envs.hybrid.cursor as envs_cursor

        env_class = getattr(envs_cursor, env_classname)
    elif "Reacher" in env_classname:
        import seads_envs.hybrid.reacher as envs_reacher

        env_class = getattr(envs_reacher, env_classname)
    elif "Jaco" in env_classname:
        import seads_envs.hybrid.jaco as envs_jaco

        env_class = getattr(envs_jaco, env_classname)

    else:
        raise ValueError

    env = env_class(
        reset_split=reset_split,
        board_size=boardsize,
        **env_kwargs,
    )
    if time_limit:
        # noinspection PyUnboundLocalVariable
        # 'max_steps' introduced in above 'if' clause
        env = TimeLimitWrapper(env, max_steps, ext_timelimit_obs)
    return env
