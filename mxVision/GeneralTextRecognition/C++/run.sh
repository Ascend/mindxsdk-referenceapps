#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
# Description: sample script for multi-thread test
# Create: 2021-06-30

set -e
build_type="Debug"

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

file_path=$(cd $(dirname $0); pwd)

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":"/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64":"/usr/local/Ascend/driver/lib64":${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"

# complie
cmake -S . -DCMAKE_BUILD_TYPE=$build_type -Bbuild
make -C ./build  -j

# run
./MultiThread
exit 0
