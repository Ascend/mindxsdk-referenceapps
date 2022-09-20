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


# 设置环境变量，注意ascend-toolkit安装路径、SDK安装路径需要进行修改
#eg：export install_path=/usr/local/Ascend/ascend-toolkit/latest

export install_path=${ascend-toolkit安装路径}/latest

export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH

export PATH=/opt/anaconda3/envs/py392/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH

export ASCEND_OPP_PATH=${install_path}/opp

export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg

export MX_SDK_HOME=${SDK安装路径}/mxVision

export LD_LIBRARY_PATH=${MX_SDK_HOME}/python:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/include:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/5.0.4/acllib/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64

export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:${MX_SDK_HOME}/python:/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:${MX_SDK_HOME}/python/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl

export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins