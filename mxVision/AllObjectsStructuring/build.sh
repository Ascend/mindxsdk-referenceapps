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

path_cur=$(cd $(dirname $0); pwd)
build_type="Release"

function prepare_path() {
    if [ -d "$1" ]; then
      rm -rf $1
    else
      echo "file $1 is not exist."
    fi
    mkdir -p $1
    cd  $1
}

function build() {
    echo ${path_cur}
    path_build=$path_cur/build
    prepare_path $path_build
    cmake -DCMAKE_BUILD_TYPE=$build_type ..
    if [ $? -ne 0 ]; then
        echo "cmake failed"
        exit -1
    fi
    make -j8
    if [ $? -ne 0 ]; then
        echo "make failed"
        exit -1
    fi
    make install
    cd ..
}

build