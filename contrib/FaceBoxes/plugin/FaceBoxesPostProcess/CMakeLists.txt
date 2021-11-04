cmake_minimum_required(VERSION 3.5.2)
project(faceboxespostprocess)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)

set(PLUGIN_NAME "faceboxespostprocess")
set(TARGET_LIBRARY ${PLUGIN_NAME})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/opensource/include)
include_directories(${PROJECT_SOURCE_DIR}/opensource/include/gstreamer-1.0)
include_directories(${PROJECT_SOURCE_DIR}/opensource/include/glib-2.0)
include_directories(${PROJECT_SOURCE_DIR}/opensource/lib/glib-2.0/include)
include_directories(${PROJECT_SOURCE_DIR}/opensource/include/opencv4)

link_directories(${PROJECT_SOURCE_DIR}/opensource/lib/)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-DENABLE_DVPP_INTERFACE)
add_library(${TARGET_LIBRARY} SHARED FaceBoxesPostProcess.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase opencv_world)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib/modelpostprocessors)
