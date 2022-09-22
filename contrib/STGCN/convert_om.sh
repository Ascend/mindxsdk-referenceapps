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

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.onnx xx"

  exit 1
fi

input_path=$1
output_om_path=$2

export install_path=/usr/local/Ascend/ascend-toolkit/5.1.RC1

export ASCEND_ATC_PATH=${install_path}/atc
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/latest/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc --framework=5 \
    --model="${input_path}" \
    --input_shape="actual_input_1:64,1,12,156"  \
    --output="${output_om_path}" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision \
    --output_type=FP32
