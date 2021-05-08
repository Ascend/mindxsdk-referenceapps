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

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":"/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64":${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"

# complie
g++ main.cpp -I "${MX_SDK_HOME}/include/" -I "${MX_SDK_HOME}/opensource/include/" -L "${MX_SDK_HOME}/lib/" \
-L "${MX_SDK_HOME}/opensource/lib/" -std=c++11 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private -fPIC -fstack-protector-all \
-g -Wl,-z,relro,-z,now,-z,noexecstack -pie -Wall -lglog -lmxbase -lstreammanager -lcpprest -lmindxsdk_protobuf -o main

# run
./main
exit 0
