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


import torch
from model.IAT_main import IAT


def pth2onnx(input_file, output_file):
    """
    Convert .pth to .onnx
    :param input_file: .pth file path of IAT
    :param output_file: .onnx file path of IAT
    :return: None
    """

    #load model
    model = IAT()
    state_dict = {k.replace('module.', ''): v for k, v in torch.load(
        input_file, map_location='cpu').items()}
    model.load_state_dict(state_dict)

    #setting
    model.eval()
    input_names = ["input_1"]
    output_names = ["output_1"]
    dummy_input = torch.randn(1, 3, 400, 600)

    #pth2onnx
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=11)


def main():
    pth2onnx('best_Epoch_lol_v1.pth', 'IAT_lol.onnx')


if __name__ == "__main__":
    main()
