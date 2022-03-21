#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

set -e

function check_index() {
    run_with_index=false
    if [ ${param_num} -gt 0 ]; then
        for param in ${params};
        do
            if [ "${param}" == "index" ]; then
                run_with_index=true
                break
            fi
        done
    fi
}

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

if [ -z "${MX_SDK_HOME}" ]; then
    echo "MX_SDK_HOME not found"
    exit -1
fi

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/opt/OpenBLAS/lib:/usr/local/lib:/usr/local/protobuf/lib:$PWD/dist/lib:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins:$PWD/dist/lib
export PYTHONPATH=${MX_SDK_HOME}/python:$PWD/dist/python:$PYTHONPATH

param_num=$#
params=$@
check_index
if [ "${run_with_index}" == true ]; then
    echo "run with index"
    python3 main.py
else
    echo "run main pipeline"
    python3 main.py -main-pipeline-only=True
fi
