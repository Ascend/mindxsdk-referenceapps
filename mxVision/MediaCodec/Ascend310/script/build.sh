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

file_path=$(cd $(dirname $0); pwd)
build_type="Debug"

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

function build_mxvision() {

    build_path=$file_path/../build
    if [ -d "$build_path" ]; then
      rm -rf $build_path
    else
      echo "file $build_path is not exist."
    fi
    mkdir -p $build_path
    cd $build_path
    cmake -DCMAKE_BUILD_TYPE=$build_type ..
    make -j
    if [ $? -ne 0 ]; then
        echo "Build Failed"
        exit -1
    fi
    cd ..
    cp ./build/mxVisionMediaCodec ./dist/
    exit 0
}

build_mxvision
exit 0