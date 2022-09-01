# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
import re
import sys
import onnx
import numpy as np

onnx_model = onnx.load(sys.argv[1])
graph = onnx_model.graph


def check_string_(re_exp, str_):
    res = re.search(re_exp, str_)
    if res:
        return True
    else:
        return False


for node in graph.initializer:
    if check_string_('.*bn.*weight', node.name):
        F = ''
        for i in range(node.dims[0]):
            F += 'f'
        value = np.array(struct.unpack(F, node.raw_data), dtype=np.float32)
        value = np.where(abs(value) > 0.01, value, 0.01)
        value = struct.pack(F, *value)
        node.raw_data = value

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, sys.argv[2])
