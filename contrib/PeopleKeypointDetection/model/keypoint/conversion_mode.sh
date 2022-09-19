#!/bin/bash

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


# 将 pth 模型转换成为.om模型


# 设置环境变量，注意install_path改用自己的

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=${install_path}/opp
export PATH=/opt/anaconda3/envs/py392/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg

# 执行，转换 3DMPPE-ROOTNET 模型

atc --framework=5 --model=3DMPPE-ROOTNET.onnx --output=3DMPPE-ROOTNET_bs1 --input_format=NCHW --input_shape="image:1,3,256,256;cam_param:1,1" --log=error --soc_version=Ascend310

# 删除 om 模型外额的所有生成文件

rm -rf kernel_meta 
rm fusion_result.json
