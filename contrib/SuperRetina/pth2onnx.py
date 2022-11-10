"""
Copyright(C) Huawei Technologies Co.,Ltd. 2022 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.onnx
from model.super_retina import SuperRetina


def pth2onnx(onnx_input, checkpoint, onnx_path, input_names="input", output_names="output", device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('The onnx path is not correct!')
        return 0

    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = SuperRetina()
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    torch.onnx.export(model, onnx_input, onnx_path, verbose=True,
    input_names=[input_names], output_names=[output_names], opset_version=11)
    print("Exporting .pth model to onnx model has been successful!")
    return 0

if __name__ == '__main__':
    CHECKPOINT = './SuperRetina.pth' # checkpoint path
    ONNX = './SuperRetina.onnx' # onnx path
    inputs = torch.randn(2, 1, 768, 768)
    pth2onnx(inputs, CHECKPOINT, ONNX)

