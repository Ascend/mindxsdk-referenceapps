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
cur_path=$(
  cd "$(dirname "$0")" || exit
  pwd
)
# env ready
env_ready=true

# execute arguments
mode='infer'
input=''
output=''

function parse_arguments() {
  echo "parsing arguments ..."
  while getopts "m:i:o:" opt; do
    case ${opt} in
    m)
      mode=$OPTARG
      ;;
    i)
      input=$OPTARG
      ;;
    o)
      output=$OPTARG
      ;;
    *)
      echo "*分支:${OPTARG}"
      ;;
    esac
  done

  # print arguments
  echo "---------------------------"
  echo "| execute arguments"
  echo "| mode: $mode"
  echo "| input: $input"
  echo "| output: $output"
  echo "---------------------------"
}

function check_env() {
  echo "checking env ..."
  # check MindXSDK env
  if [ ! "${MX_SDK_HOME}" ]; then
    env_ready=false
    echo "please set MX_SDK_HOME path into env."
  else
    echo "MX_SDK_HOME set as ${MX_SDK_HOME}, ready."
  fi
}

function export_env() {
  export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}
  export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
  export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
  export PYTHONPATH=${MX_SDK_HOME}/python

  echo "export env success."
}

function execute() {
  if [ "${mode}" == 'infer' ]; then
    python3.7 main.py "${input}" "${output}"
  elif [ "${mode}" == 'evaluate' ]; then
    python3.7 evaluate.py
  fi
}

function run() {
  echo -e "\ncurrent dir: $cur_path"
  # parse arguments
  parse_arguments "$@"

  # check environment
  check_env
  if [ "${env_ready}" == false ]; then
    echo "please set env first."
    exit 0
  fi

  # export environment
  export_env

  echo "---------------------------"
  echo "prepare to execute program."
  echo -e "---------------------------\n"

  # execute
  execute
}

run "$@"
exit 0
