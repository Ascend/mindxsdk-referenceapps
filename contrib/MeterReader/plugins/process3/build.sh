#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

# 使用cmake编译插件

mkdir build
mkdir lib
cd lib
mkdir plugins
cd ..

cd build
cmake ..
make
cd ..

# 得到的插件位于../lib/plugins
# 复制到lib中

cp lib/plugins/libmxpi_sampleplugin.so "$MX_SDK_HOME/lib/plugins"
chmod 440 "$MX_SDK_HOME/lib/plugins/libmxpi_sampleplugin.so"



