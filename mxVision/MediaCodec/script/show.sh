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

log_path=${file_path}/../logs
num=$(ls ${log_path} | grep .log | grep output | wc -w)

for((i=0;(i<${num});i++));
do
    cat ${log_path}/output${i}.log | tail -n 100 | grep "mxpi_videoencoder" | grep "fps" | tail -n 2
done

for((i=0;(i<${num});i++));
do
    cat ${log_path}/output${i}.log | tail -n 100 | grep "stream size" | tail -n 2
done

exit 0