cmake_minimum_required(VERSION 3.10)
project(mxBase_sentiment_analysis)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")
include_directories(./SentimentAnalysis)
file(GLOB_RECURSE SentimentAnalysis ${PROJECT_SOURCE_DIR}/SentimentAnalysis/*cpp)
include_directories(./test)
file(GLOB_RECURSE Test ${PROJECT_SOURCE_DIR}/test/*cpp)
set(TARGET mxBase_text_classification)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
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

add_executable(mxBase_sentiment_analysis main.cpp ${SentimentAnalysis} ${Test})

target_link_libraries(mxBase_sentiment_analysis
        glog
        mxbase
        cpprest
        opencv_world
        )