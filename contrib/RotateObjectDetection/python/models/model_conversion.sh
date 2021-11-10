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

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
# export DUMP_GE_GRAPH=1


# 执行，转换YOLOv3模型
# Execute, transform YOLOv3 model.

atc --model=./YOLOv5_DOTAv1.5_OBB_1024_1024.onnx --framework=5 --output=./YOLOv5_DOTAv1.5_OBB_1024_1024 --input_format=NCHW --log=info --soc_version=Ascend310 --insert_op_conf=./aipp_yolov5_1024_1024.aippconfig --input_shape="images:1,3,1024,1024"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。