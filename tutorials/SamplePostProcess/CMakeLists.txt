# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
cmake_minimum_required(VERSION 3.5.2)

project(mxpi_sampleplugin)

#该语句中%MX_SDK_HOME%根据实际SDK安装位置修改，可通过在终端运行env命令查看
set(MX_SDK_HOME %MX_SDK_HOME%)

add_subdirectory("./mxVision/C++")
add_subdirectory("./src/samplepostprocess")
add_subdirectory("./src/yolov3")