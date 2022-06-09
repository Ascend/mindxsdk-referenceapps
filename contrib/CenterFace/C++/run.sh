#!/bin/bash

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
# env ready
env_ready=true

function check_env() {

    # check MindXSDK env
    if [ ! "${MX_SDK_HOME}" ]; then
      env_ready=false
      echo "please set MX_SDK_HOME path into env."
    else
      echo "MX_SDK_HOME set as ${MX_SDK_HOME}, ready."
    fi


}

function export_env() {
    export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
    export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
    export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

}

echo "current dir: $cur_path"

check_env
if [ "${env_ready}" == false ]; then
  echo "please set env first."
  exit 0
fi

export_env
echo "export env success."
echo "prepare to execute main program."

# check file
if [ ! -f "${cur_path}/Main" ]; then
  echo "Main not exist, please build first."
else
  # execute
  # set your picture file_path
  fileName=$1
  if [ -f $fileName ]
  then
    ./Main $fileName
  else
    if [ -d $fileName ]
    then
      for file in ./${fileName}/*
      do
          if test -f $file
          then
            echo $file
            ./Main $file || true
          fi
      done
    else
      echo "error: ${cur_path}/${fileName} not exist."
    fi
  fi
fi

exit 0
