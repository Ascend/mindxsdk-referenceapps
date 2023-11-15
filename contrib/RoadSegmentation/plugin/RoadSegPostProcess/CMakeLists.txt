cmake_minimum_required(VERSION 3.14.1)
project(roadsegpostprocess)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(PLUGIN_NAME "roadsegpostprocess")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/plugins)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)
include_directories(

    $ENV{MX_SDK_HOME}/opensource/include/

    $ENV{MX_SDK_HOME}/opensource/include/opencv4

    ${INC_PATH}/acllib/include/

    ../inc/

)

link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-DENABLE_DVPP_INTERFACE)

add_library(${TARGET_LIBRARY} SHARED MxpiRoadSegPostProcess.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)

target_link_libraries(${TARGET_LIBRARY} mxpidatatype plugintoolkit mxbase mindxsdk_protobuf)

install(TARGETS ${PLUGIN_NAME} PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ)

