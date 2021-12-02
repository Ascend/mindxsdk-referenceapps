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

set -e

# curr path
cur_path=$(cd "$(dirname "$0")" || exit; pwd)

# build type
build_type="Release"

function prepare_env()
{
   export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/:$LD_LIBRARY_PATH
}

function prepare_path() {
    if [ -d "$1" ]; then
      rm -rf "$1"
      echo "dir $1 exist, erase it and recreate."
    else
      echo "dir $1 is not exist."
    fi
    mkdir -p "$1"
    cd  "$1"
}

function build() {
    echo "current dir: $cur_path"
    prepare_env
    path_build=${cur_path}/build
    prepare_path "$path_build"

    cmake -DCMAKE_BUILD_TYPE=$build_type ..
    # shellcheck disable=SC2181
    if [ $? -ne 0 ]; then
        echo "cmake failed"
        exit 0
    fi
    make -j8
    # shellcheck disable=SC2181
    if [ $? -ne 0 ]; then
        echo "make failed"
        exit 0
    fi
    cd ..
}

build