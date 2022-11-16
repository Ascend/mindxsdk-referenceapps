# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mindspore import nn


class PatchCore(nn.Cell):

    def __init__(self, network, layer="layer2"):
        super(PatchCore, self).__init__()
        self.network = network
        self.layer = layer

    def construct(self, x):
        if self.layer == "layer2":
            layer = self.network(x)[0]
        else:
            layer = self.network(x)[1]

        return layer
