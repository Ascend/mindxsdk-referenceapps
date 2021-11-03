#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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
import onnx
import torch
from torch.onnx import OperatorExportTypes


def transform_pt_to_onnx(model_type):
    """
        turn *.pth to *.onxx
        Args:
            model_type：turn DANet or Deeplabv3
        Returns:
            null
        Output:
            the model of *.onxx in the models directory
    """
    pretrained_path = ''
    onnx_output_path = ''

    if model_type == 'DANet':
        pretrained_path = "models/84_DANet_0.8081.pth"
        onnx_output_path = "models/DANet.onnx"
    elif model_type == 'Deeplabv3':
        pretrained_path = "models/37_Deeplabv3+_0.8063.pth"
        onnx_output_path = "models/Deeplabv3.onnx"

    # load model
    model = torch.load(pretrained_path, map_location='cpu').module
    # set the model to inference mode
    model.eval()

    img = torch.rand(1, 3, 256, 256)
    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(model,  # model being run
                      img,  # model input (or a tuple for multiple inputs)
                      onnx_output_path,  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      operator_export_type=OperatorExportTypes.ONNX,
                      input_names=input_names,  # the model's input names
                      output_names=output_names)  # the model's output names)

    # check model
    onnx_model = onnx.load(onnx_output_path)
    print('check: ', onnx.checker.check_model(onnx_model))


if __name__ == '__main__':
    transform_pt_to_onnx('DANet')
    transform_pt_to_onnx('Deeplabv3')
