#!/bin/bash

# Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.
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


# 该脚本用来将 pth 模型文件转换成.om模型文件
# This is used to convert pth model file to .om model file.

# 执行，转换 Openpose 模型
# Execute, transform Openpose model.

atc --model=./simplified_560_openpose_pytorch.onnx --framework=5 --output=openpose_pytorch_560 --soc_version=Ascend310B1 --input_shape="data:1, 3, 560, 560" --input_format=NCHW --insert_op_conf=./insert_op.cfg

# 删除除 om 模型外额外生成的文件
# Remove miscellaneous

rm fusion_result.json
rm -rf kernel_meta 
