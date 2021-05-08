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

#需要生成的多少路pipelibe
channel_nums=xxx

#每路pipeline的rtsp流地址，数组长度跟${channel_nums}一致，请确保rtsp地址存在。
#若使用live555推流工具构架rtsp流，rstp流格式为rtsp://${ip_addres}:${port}/${h264_file}
#${ip_addres}：起流的机器ip地址。如果是本地起流可以设置为127.0.0.1；如果是其他机器起流，那需要配置该台机器的ip地址
#${port}：可使用的rtsp流的端口
#${h264_file}:需要推流的h264视频文件，一般都是以.264结尾的文件
rtsp_array=(xxx xxx xxx)

#配置pipeline运行的npu编号
device_id=xxx

#输出图像尺寸. CIF(height: 288 width: 352),D1(height: 480 width: 720)
height=xxx
width=xxx

#是否打印转码的帧率. 0：不打印，1：打印
fps=xxx

#I帧间隔.一般设置视频帧率大小，25或者30
i_frame_interval=xxx


pipeline_path=$file_path/../pipeline
rm -rf $pipeline_path/test?.pipeline
for((i=0;(i<${channel_nums});i++));
do
    cp $pipeline_path/test.pipeline $pipeline_path/test${i}.pipeline

    sed -i "s/\"mxpi_rtspsrcxxx\"/\"mxpi_rtspsrc${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"mxpi_videodecoderxxx\"/\"mxpi_videodecoder${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"mxpi_imageresizexxx\"/\"mxpi_imageresize${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"mxpi_videoencoderxxx\"/\"mxpi_videoencoder${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"fakesinkxxx\"/\"fakesink${i}\"/g" $pipeline_path/test${i}.pipeline

    sed -i "s/\"deviceId\"\: \"xxx\"/\"deviceId\"\: \"${device_id}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"vdecChannelId\"\: \"xxx\"/\"vdecChannelId\"\: \"${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"imageHeight\"\: \"xxx\"/\"imageHeight\"\: \"${height}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"imageWidth\"\: \"xxx\"/\"imageWidth\"\: \"${width}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"resizeHeight\"\: \"xxx\"/\"resizeHeight\"\: \"${height}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"resizeWidth\"\: \"xxx\"/\"resizeWidth\"\: \"${width}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"channelId\"\: \"xxx\"/\"channelId\"\: \"${i}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"fps\"\: \"xxx\"/\"fps\"\: \"${fps}\"/g" $pipeline_path/test${i}.pipeline
    sed -i "s/\"iFrameInterval\"\: \"xxx\"/\"iFrameInterval\"\: \"${i_frame_interval}\"/g" $pipeline_path/test${i}.pipeline

    rtsp=$(echo ${rtsp_array[${i}]} | sed -e 's/\//\\\//g' | sed -e 's/\:/\\\:/g' |  sed -e 's/\_/\\\_/g')
    sed -i "s/\"rtspUrl\"\: \"xxx\"/\"rtspUrl\"\: \"${rtsp}\"/g" $pipeline_path/test${i}.pipeline
done
exit 0
