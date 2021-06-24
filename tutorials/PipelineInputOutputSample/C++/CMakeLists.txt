cmake_minimum_required(VERSION 3.10)
project(IOsample)

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(CMAKE_CXX_STANDARD 14)
set(MX_SDK_HOME /usr/local/Ascend/mxVision)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)

link_directories(${MX_SDK_HOME}/lib)
link_directories(
        ${MX_SDK_HOME}/opensource/lib
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/

        )

add_executable(IOsample main.cpp)

target_link_libraries(IOsample
        glog
        mxbase
        plugintoolkit
        mxpidatatype
        streammanager
        cpprest
        mindxsdk_protobuf
        opencv_world)



