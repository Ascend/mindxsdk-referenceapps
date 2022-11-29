# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
import utils
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops



class SSlayer(nn.Cell):
    def __init__(self, requires_grad=False):
        super().__init__()

        weight = utils.get_3dgauss()  # [16,16]
        weight = weight.expand_dims(0).expand_dims(0)  # [1,1,16,16]
        weight = ops.concat([weight, weight, weight], axis=1)   # [1,3,16,16]
        weight = ops.concat([weight, weight, weight], axis=0)   # [3,3,16,16]
        conv_module = nn.Conv2d(in_channels=3, out_channels=3, 
            kernel_size=16, stride=16, padding=0, 
            has_bias=False, weight_init=weight)
        self.conv_module = nn.SequentialCell([
            conv_module
        ])

        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False # each kernel is fixed to gauss weight

    def construct(self, x):
        x = mnp.tile(x, (1, 1, 1, 1))
        x = self.conv_module(x)
        return x  # return x for visualization


if __name__ == '__main':
    net = SSlayer()