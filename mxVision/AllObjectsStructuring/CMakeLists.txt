cmake_minimum_required(VERSION 3.5.1)
project(AllObjectsStructuring)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(CMAKE_CXX_STANDARD 11)
set(PROJECT_DIR ${PROJECT_SOURCE_DIR})

if(NOT DEFINED ENV{MX_SDK_HOME})
    message(FATAL_ERROR "MX_SDK_HOME is not defined, please set it first.")
else()
    set(MX_SDK_HOME $ENV{MX_SDK_HOME})
    message("MX_SDK_HOME=$ENV{MX_SDK_HOME}")
endif()

set(ENV{LD_LIBRARY_PATH} ${MX_SDK_HOME}/opensource/lib:$LD_LIBRARY_PATH)

include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)

add_subdirectory(Proto)
add_subdirectory(plugins/MpObjectSelection)
add_subdirectory(plugins/MxpiFaceSelection)
add_subdirectory(plugins/MxpiSkipFrame)
add_subdirectory(plugins/MxpiFrameAlign)