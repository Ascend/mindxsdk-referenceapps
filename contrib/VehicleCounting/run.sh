#!/bin/bash
# Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export MX_SDK_HOME={SDK实际安装路径}
export FFMPEG_PATH={ffmpeg实际安装路径}
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/:/usr/local/python3.7.5/lib:${FFMPEG_PATH}/lib:${LD_LIBRARY_PATH}

./stream_pull_test