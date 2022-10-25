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

import io
import torch
import torch.onnx
import torch.nn.parallel
from mobilenet_v2_tsm import MobileNetV2


def pth_to_onnx(input_shape, checkpoint, onnx_path):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    input_names = []
    input_shapes = {}
    for index, torch_input in enumerate(input_shape):
        name = "i" + str(index)
        input_names.append(name)
        input_shapes[name] = torch_input.shape
    buffer = io.BytesIO()
    net = MobileNetV2(n_class=27)
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint)
    net.eval()
    torch.onnx.export(net, input_shape, onnx_path, input_names=input_names, 
                      output_names=["o" + str(i) for i in range(len(input_shape))], opset_version=10)
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    PTH = './mobilenetv2_jester_online.pth.tar'
    ONNX_PATH = './jester.onnx'
    INPUT_SHAPE = (torch.rand(1, 3, 224, 224),
                    torch.rand([1, 3, 56, 56]),
                    torch.rand([1, 4, 28, 28]),
                    torch.rand([1, 4, 28, 28]),
                    torch.rand([1, 8, 14, 14]),
                    torch.rand([1, 8, 14, 14]),
                    torch.rand([1, 8, 14, 14]),
                    torch.rand([1, 12, 14, 14]),
                    torch.rand([1, 12, 14, 14]),
                    torch.rand([1, 20, 7, 7]),
                    torch.rand([1, 20, 7, 7]))
    pth_to_onnx(INPUT_SHAPE, PTH, ONNX_PATH)

