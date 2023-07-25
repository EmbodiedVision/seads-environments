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

from setuptools import find_packages, setup

packages = find_packages(".", include="seads_envs*")

setup(
    name="seads-environments",
    description="Physically embedded single-player board game environments, used to evaluate the SEADS agent.",
    author="Jan Achterhold",
    author_email="jan.achterhold@tuebingen.mpg.de",
    url="https://github.com/EmbodiedVision/seads-environments/",
    packages=packages,
    package_data={p: ["*", "**/*"] for p in packages},  # include all package data
    install_requires=[
        "dm-control==0.0.364896371",
        "dm-env==1.5",
        "gym==0.19.0",
        "h5py",
        "joblib",
        "jupyter",
        "matplotlib",
        "mujoco-py==2.0.2.13",
        "numpy",
        "pandas",
        "Pillow",
        "scipy",
        "tabulate",
        "tqdm",
    ],
    # Install with pip install .[develop] to install development packages
    extras_require={
        "develop": [
            "black==20.8b1",
            "click==8.0.1",
            "isort==5.7.0",
            "pre-commit==2.19.0",
        ],
    },
)
