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

# This is used to convert onnx model file to .om model file.
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export Home="./path"
# Home is set to the path where the model is located

# Execute, transform PFLD model.
atc --framework=5 --model="${Home}"/pfld_106.onnx --output="${Home}"/pfld_106 --input_format=NCHW --insert_op_conf=./aipp_pfld_112_112.aippconfig  --input_shape="input_1:1,3,112,112" --log=debug --soc_version=Ascend310
# --model is the path where onnx is located. 
# --output is the path where the output of the converted model is located