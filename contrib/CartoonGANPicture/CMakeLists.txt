cmake_minimum_required(VERSION 3.10)
project(CartoonGAN_picture)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME  "$ENV{MX_SDK_HOME}")

include_directories(./CartoonGANPicture)
include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
        ${MX_SDK_HOME}/opensource/include/boost/
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
)

add_executable(CartoonGAN_picture main.cpp CartoonGANPicture/CartoonGANPicture.cpp)
target_link_libraries(CartoonGAN_picture
        glog
        mxbase
        cpprest
        opencv_world
        yolov3postprocess
        boost_filesystem
        stdc++fs
        )
