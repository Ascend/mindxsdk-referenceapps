#!/bin/bash

CUR_PATH=$(cd "$(dirname "$0")"; pwd)

echo $CUR_PATH

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":"/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64":"/usr/local/Ascend/driver/lib64/ ":${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"

./sample