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

current_folder="$( cd "$(dirname "$0")" ;pwd -P )"

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# complie
SAMPLE_FOLDER=(
	plugins/FaceDetectPostProcessor/
	plugins/KeyPointPostProcessor/
  C++/
)

#echo "sample_forder success"
err_flag=0
for sample in "${SAMPLE_FOLDER[@]}";do
    echo "for :"
    echo "${current_folder}/${sample}"
    cd "${current_folder}/${sample}"
    bash build.sh || {
        echo -e "Failed to build ${sample}"
		err_flag=1
    echo "over"
    }
done


if [ ${err_flag} -eq 1 ]; then
	exit 1
fi

exit 0