cmake_minimum_required(VERSION 3.5.1)
project(MediaCodec310P)

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED ENV{MX_SDK_HOME})
    message(FATAL_ERROR "MX_SDK_HOME is not defined, please set it first.")
else()
    set(MX_SDK_HOME $ENV{MX_SDK_HOME})
    message("MX_SDK_HOME=$ENV{MX_SDK_HOME}")
endif()

add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
set(TARGET_NAME mediaCodec310P)

include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)

link_directories(/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64)

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} mxbase streammanager cpprest glog pthread)
