# Copyright(c) Huawei Technologies Co. Ltd. 2019-2021. All right reserved.
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.2)

# project information
project(MindX_SDK_Sample)

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
)

add_executable(main main_sequential.cpp)
target_link_libraries(main
        glog
        mxbase
        mxpidatatype
        plugintoolkit
        streammanager
        cpprest
        mindxsdk_protobuf
)
