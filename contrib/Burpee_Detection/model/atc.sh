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

# atc environment
source ../envs/env.sh
source ../envs/atc_env.sh

# atc transform model
atc \
  --model=./burpee_detection.onnx \
  --framework=5 \
  --output=./burpee_detection \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640"  \
  --out_nodes="Transpose_213:0;Transpose_262:0;Transpose_311:0"  \
  --enable_small_channel=1 \
  --insert_op_conf=./aipp_yolov5.cfg \
  --soc_version=Ascend310 \
  --log=info
  
