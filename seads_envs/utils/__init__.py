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


class PatchedFunctionContext:
    def __init__(self, module, fcn_name, fcn_replacement):
        self.module = module
        self.fcn_name = fcn_name
        self.fcn_replacement = fcn_replacement
        self.original_function = None

    def __enter__(self):
        self.original_function = getattr(self.module, self.fcn_name)
        setattr(self.module, self.fcn_name, self.fcn_replacement)

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.module, self.fcn_name, self.original_function)
