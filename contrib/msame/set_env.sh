#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
# Description: SDK uninstallation tool.
# Author: MindX SDK
# Create: 2020
# History: NA

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
  sdk_path=$(cd $(dirname $path); pwd )

  if [[ -f "$sdk_path"/filelist.txt ]] && [[ -f "$sdk_path"/version.info ]];then
    export MX_SDK_HOME="$sdk_path"
    export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
    export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"
    export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors":"${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":${LD_LIBRARY_PATH}
    export PYTHONPATH=${MX_SDK_HOME}/python:$PYTHONPATH
  else
    echo "The package is incomplete, please check it."
  fi
else
  echo "There is no 'set_env.sh' to import"
fi

