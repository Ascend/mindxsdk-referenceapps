cmake_minimum_required(VERSION 3.10)
project(MultiChannelVideoDetection)

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(OUTPUT_NAME "multiChannelVideoReasoner")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
set(FFMPEG_HOME {ffmpeg实际安装路径})

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
        ${FFMPEG_HOME}/include
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        ${FFMPEG_HOME}/lib
)

add_executable(${OUTPUT_NAME} main.cpp
        StreamPuller/StreamPuller.cpp
        VideoDecoder/VideoDecoder.cpp
        ImageResizer/ImageResizer.cpp
        YoloDetector/YoloDetector.cpp
        MultiChannelVideoReasoner/MultiChannelVideoReasoner.cpp
        Util/PerformanceMonitor/PerformanceMonitor.cpp Util/Util.cpp)
target_link_libraries(${OUTPUT_NAME}
        avcodec
        avdevice
        avfilter
        avformat
        avcodec
        avutil
        swscale
        swresample
        glog
        mxbase
        cpprest
        opencv_world
        pthread m
        yolov3postprocess
        )