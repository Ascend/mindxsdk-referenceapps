#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# Description: run.sh.
# Author: MindX SDK
# Create: 2022
# History:NA

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)

info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

. /usr/local/Ascend/ascend-toolkit/set_env.sh

# compile
cmake -S . -Bbuild
make -C ./build -j

# run
./build/mediaCodec310P

exit 0