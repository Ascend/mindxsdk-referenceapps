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

# 该脚本用来将模型文件转换成.om模型文件的环境变量

# set ATC environment variate
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:$PATH
export PYTHONPATH=${install_path}/arm64-linux/atc/python/site-packages:${install_path}/arm64-linux/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/arm64-linux/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/arm64-linux/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
# insert sucessfully
echo "insert success!"
