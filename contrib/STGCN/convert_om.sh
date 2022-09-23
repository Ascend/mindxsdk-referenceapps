# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

input_path=$1
output_om_path=$2

/home/chongqin1/Ascend/ascend-toolkit/5.1.RC1/atc/bin/atc \
    --framework=5 \
    --model="${input_path}" \
    --input_shape="input:64,1,12,156"  \
    --output="${output_om_path}" \
    --enable_small_channel=1 \
    --log=error \
    --input_format=NCHW \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision \
    --output_type=FP32

