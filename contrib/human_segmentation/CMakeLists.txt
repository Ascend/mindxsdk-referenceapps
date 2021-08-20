cmake_minimum_required(VERSION 3.10)
project(C++_human_segmentation)

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME /{安装位置}/MindXSDK/mxVision)
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

add_executable(human_segmentation main.cpp)
target_link_libraries(human_segmentation
        glog
        mxbase
        mxpidatatype
        plugintoolkit
        streammanager
        cpprest
        mindxsdk_protobuf
        opencv_world
)
