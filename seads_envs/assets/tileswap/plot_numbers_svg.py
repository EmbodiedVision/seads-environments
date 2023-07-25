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


import xml.etree.ElementTree as element_tree

from cairosvg import svg2png

tree = element_tree.parse("chip_base.svg")
root = tree.getroot()

text_content_elems = root.findall(".//*[@id='chip_text_content']")
assert len(text_content_elems) == 1
text_content_elem = text_content_elems[0]

for n in range(10):
    text_content_elem.text = str(n)
    new_svg = element_tree.tostring(root, encoding="utf8", method="xml")
    svg2png(
        bytestring=new_svg, write_to=f"{n}.png", output_width=100, output_height=100
    )
