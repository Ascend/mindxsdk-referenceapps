# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
cmake_minimum_required(VERSION 3.5.2)

project(mxpi_samplecustomproto)

set(MX_SDK_HOME /usr/local/Ascend/mxVision)

add_subdirectory("./mxVision/C++")
add_subdirectory("./mindx_sdk_plugin")
add_subdirectory("./proto")