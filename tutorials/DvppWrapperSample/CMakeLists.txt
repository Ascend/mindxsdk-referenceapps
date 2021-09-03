# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(dvpp_warpper_sample)

# Compile options
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -Wall)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")

set(MX_SDK_HOME /home/kongchibin/MindXSDK/mxVision)

# Header path
include_directories(
    ${MX_SDK_HOME}/include/
    ${MX_SDK_HOME}/opensource/include/
    ${MX_SDK_HOME}/opensource/include/opencv4
)

# add host lib path
link_directories(
    ${MX_SDK_HOME}/lib/
    ${MX_SDK_HOME}/opensource/lib/
    ${MX_SDK_HOME}/opensource/lib64/
    ${MX_SDK_HOME}/lib/modelpostprocessors
    ${MX_SDK_HOME}/include/MxBase/postprocess/include
)

add_executable(DvppWrapperSample main.cpp)

target_link_libraries(DvppWrapperSample glog mxbase streammanager cpprest mindxsdk_protobuf)

install(TARGETS DvppWrapperSample DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
