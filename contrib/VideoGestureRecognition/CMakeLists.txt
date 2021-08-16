cmake_minimum_required(VERSION 3.10)
project(MultiChannelVideoDetection)

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

set(MX_SDK_HOME /home/cqu_liu9/MindXSDK/mxVision)
set(FFMPEG_PATH /home/cqu_liu9/ascend_ddk/x86)

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
        ${FFMPEG_PATH}/include
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        ${FFMPEG_PATH}/lib
)

add_executable(sample main.cpp
        Util/Util.h
        BlockingQueue/BlockingQueue.h
        StreamPuller/StreamPuller.cpp StreamPuller/StreamPuller.h
        VideoDecoder/VideoDecoder.cpp VideoDecoder/VideoDecoder.h
        ImageResizer/ImageResizer.cpp ImageResizer/ImageResizer.h
        ResnetDetector/ResnetDetector.cpp ResnetDetector/ResnetDetector.h
        MultiChannelVideoReasoner/MultiChannelVideoReasoner.cpp MultiChannelVideoReasoner/MultiChannelVideoReasoner.h
        FrameSkippingSampling/FrameSkippingSampling.cpp FrameSkippingSampling/FrameSkippingSampling.h)
target_link_libraries(sample
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
        resnet50postprocess
        )