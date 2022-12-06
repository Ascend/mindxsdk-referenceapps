cmake_minimum_required(VERSION 3.10)
project(mediacodecV2)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set(TARGET mediacodecV2)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all )
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME ${SDK-path})

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${FFmpeg安装路径}/include
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        ${ascend-toolkit-path}/ascend-toolkit/latest/acllib/lib64
        ${ascend-toolkit-path}/driver/lib64/
        ${FFmpeg安装路径}/lib
)


add_executable(mediacodecV2 mainV2.cpp MediaCodecV2.cpp)
target_link_libraries(mediacodecV2
        glog
        mxbase
        cpprest
        opencv_world
        pthread
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale
        )
