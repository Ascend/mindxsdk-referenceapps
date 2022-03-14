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


# 该脚本用来将 onnx 模型文件转换成.om模型文件
# This is used to convert onnx model file to .om model file.


# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换 EfficientDet-d2 模型
# Execute, transform EfficientDet-d2 model.

atc --model=../onnx-models/simplified-efficient-det-d2-mindxsdk-order.onnx --framework=5 --output=../efficient-det-d2-mindxsdk-order --soc_version=Ascend310 --input_shape="input:1, 3, 768, 768" --input_format=NCHW --output_type=FP32 --out_nodes='Concat_15356:0;Sigmoid_17693:0' --log=error --insert_op_conf=../aipp-configs/insert_op_d2.cfg

# 删除除 om 模型外额外生成的文件
# Remove miscellaneous

rm fusion_result.json
rm -rf kernel_meta 
