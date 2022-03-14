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

RTSP_URL="${RTSP_URL}"

cat label.txt|
while read line
do
    sfile=`echo ${line%% *}`
    str=`echo ${line#* }`
    label1=`echo ${str%% *}`
    frame=`echo ${line##* }`
    echo $sfile
    python3 test.py \
        --url_video ${RTSP_URL}$sfile \
        --label $label1 \
        --frame_num  $frame
done

python3 evaluate.py