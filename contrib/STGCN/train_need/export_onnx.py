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

import torch
import torch.nn
import onnx

model = torch.load('model/save/sz-taxis/stgcn_sym_norm_lap_45_mins.pth')
input_names = ['input']
output_names = ['output']

x = torch.randn(64, 1, 12, 156, device='cpu')

torch.onnx.export(model, x, 'stgcn10.onnx',\
 opset_version = 12, input_names=input_names, \
 output_names=output_names, verbose='True')
