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


# 该脚本用来将pb模型文件转换成.om模型文件
# This is used to convert pb model file to .om model file.


# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

. /usr/local/Ascend/ascend-toolkit/set_env.sh   # The path where Ascend-cann-toolkit is located


atc --model=./face_mask_detection.pb --framework=3 --output=./aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="data_1:1,260,260,3" --input_format=NHWC --insert_op_conf=./face_mask.aippconfig