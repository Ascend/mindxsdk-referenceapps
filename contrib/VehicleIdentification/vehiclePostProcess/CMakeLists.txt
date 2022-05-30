cmake_minimum_required(VERSION 3.5.2)
project(vehiclepostprocess)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(PLUGIN_NAME "vehiclepostprocess")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib)

include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories(${MX_SDK_HOME}/opensource/lib)
link_directories(${MX_SDK_HOME}/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-DENABLE_DVPP_INTERFACE)
add_library(${TARGET_LIBRARY} SHARED VehiclePostProcess.cpp VehiclePostProcess.h)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0 glog)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase)
#target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)