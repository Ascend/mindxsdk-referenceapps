cmake_minimum_required(VERSION 3.10)
project(mxBase_text_classification)

set(CMAKE_CXX_STANDARD 14)
include_directories(./BertClassification)
file(GLOB_RECURSE BertClassification ${PROJECT_SOURCE_DIR}/BertClassification/*cpp)
include_directories(./test)
file(GLOB_RECURSE Test ${PROJECT_SOURCE_DIR}/test/*cpp)
set(TARGET mxBase_text_classification)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME ${SDK目录})
include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
        /usr/local/Ascend/driver/lib64/
)

add_executable(mxBase_text_classification main.cpp ${BertClassification} ${Test})

target_link_libraries(mxBase_text_classification
        glog
        mxbase
        cpprest
        opencv_world
        )