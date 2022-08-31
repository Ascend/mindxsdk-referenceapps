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


# 提示，确保先运行set_env.sh设置完成了环境变量


# 执行，转换 yolov3 模型

atc --model=yolov3_tf.pb --framework=3 --output=yolov3_tf_aipp  --input_format=NHWC --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --insert_op_conf=yolov3_tf_aipp.cfg --log=info

# 删除 om 模型外额的所有生成文件

rm -rf kernel_meta 
rm fusion_result.json