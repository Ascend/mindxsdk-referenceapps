cmake_minimum_required(VERSION 3.5.0)
project(mxplugins)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(PLUGIN_NAME "DBPostProcess")
set(TARGET_LIBRARY ${PLUGIN_NAME})

if(NOT DEFINED ENV{MX_SDK_HOME})
    message(FATAL_ERROR "please define environment variable:MX_SDK_HOME")
endif()

add_definitions(-D ENABLE_POST_PROCESS_INSTANCE)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../Clipper)
include_directories($ENV{MX_SDK_HOME}/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories($ENV{MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories($ENV{MX_SDK_HOME}/opensource/lib/glib-2.0/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)
include_directories($ENV{MX_SDK_HOME}/opensource/include/google)

link_directories($ENV{MX_SDK_HOME}/opensource/lib)
link_directories(${PROJECT_SOURCE_DIR}/../../lib)
link_directories($ENV{MX_SDK_HOME}/lib)

add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall -Wno-deprecated-declarations)
add_definitions(-DENABLE_DVPP_INTERFACE -Dgoogle=mindxsdk_private)
add_library(${TARGET_LIBRARY} SHARED DBPostProcess.cpp)

target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)
target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0
        gmodule-2.0 mxpidatatype plugintoolkit mxbase opencv_world clipper)
		
install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../../lib/)
