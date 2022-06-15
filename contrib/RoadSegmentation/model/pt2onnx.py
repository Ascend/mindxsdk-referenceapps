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
import os
import torch.onnx

# path to pt file
MODEL_DIR = './model.pt'
modelPath = os.path.join(MODEL_DIR)
model = torch.load(modelPath, map_location=torch.device("cpu"))
model.eval()
inputNames = ["image"]
# 输出节点名
outputNames = ["class"]
dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
dummy_input = torch.randn(1, 3, 224, 224)
# verbose=True，支持打印onnx节点和对应的PyTorch代码行
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "Road.onnx", input_names=inputNames, dynamic_axes=dynamic_axes,
                  output_names=outputNames, opset_version=11, verbose=True)
