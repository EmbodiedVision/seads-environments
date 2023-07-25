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

import math
from itertools import product

import matplotlib.pyplot as plt

__all__ = ["plot_2d_trajectory", "plot_skill_episodes"]


def plot_2d_trajectory(
    positions,
    activations,
    last_is_done,
    ax,
    plot_done=True,
    scatter_kwargs=None,
):
    if scatter_kwargs is None:
        scatter_kwargs = {}

    lp = ax.plot(positions[:, 0], positions[:, 1], alpha=0.5)
    ax.scatter(
        positions[activations, 0],
        positions[activations, 1],
        s=2,
        color="black",
        marker="x",
        **scatter_kwargs,
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        s=2,
        color=lp[0].get_color(),
        alpha=0.5,
        marker="o",
    )
    if plot_done and last_is_done:
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color="green",
            marker="x",
            zorder=10,
            **scatter_kwargs,
        )


def plot_skill_episodes(
    env,
    episode_list,
    num_skills=None,
    subplots_kwargs=None,
    return_ax=False,
    ax_arr=None,
    plot_done=True,
    scatter_kwargs=None,
):
    if num_skills is None:
        num_skills = env.n_groundtruth_skills

    n_rows = math.ceil(num_skills ** 0.5)
    n_cols = math.ceil(num_skills ** 0.5)

    if subplots_kwargs is None:
        subplots_kwargs = {"figsize": (2 * n_cols, 2 * n_rows)}

    if ax_arr is None:
        fig, ax_arr = plt.subplots(nrows=n_rows, ncols=n_cols, **subplots_kwargs)
    else:
        fig = None

    for row_idx, col_idx in product(range(n_rows), range(n_cols)):
        symbolic_action = row_idx * n_cols + col_idx
        filtered_episodes = [
            e for e in episode_list if e[0]["symbolic_action"] == symbolic_action
        ]
        ax = ax_arr[row_idx, col_idx]
        env.plot_board(ax)
        for episode in filtered_episodes:
            env.plot_trajectory(
                episode,
                ax,
                plot_done=plot_done,
                scatter_kwargs=scatter_kwargs,
            )
        ax.set_title(f"k={symbolic_action}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    if return_ax:
        return fig, ax_arr
    else:
        return fig
