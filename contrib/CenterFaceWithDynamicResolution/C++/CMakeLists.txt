cmake_minimum_required(VERSION 3.10)
project(CenterFace)
add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
set(CMAKE_BUILD_TYPE Debug)
set(MX_SDK_HOME $ENV{MX_SDK_HOME})

if (NOT DEFINED ENV{MX_SDK_HOME})
    string(REGEX REPLACE "(.*)/(.*)/(.*)/(.*)" "\\1" MX_SDK_HOME  ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "set default MX_SDK_HOME: ${MX_SDK_HOME}")
else ()
    message(STATUS "env MX_SDK_HOME: ${MX_SDK_HOME}")
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")


include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        /usr/local/Ascend/ascend-toolkit/latest/acllib/include
)
link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ../plugins/lib/
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
)
add_executable(Main main.cpp )
target_link_libraries(Main
        glog
        mxbase
        mxpidatatype
        plugintoolkit
        streammanager
        cpprest
        mindxsdk_protobuf
        opencv_world
        )