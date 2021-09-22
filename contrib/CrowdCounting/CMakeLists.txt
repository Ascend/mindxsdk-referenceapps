cmake_minimum_required(VERSION 3.10)
project(crowd_counting)

include_directories(./CrowdCountPostProcess)
include_directories(./CrowdCount)
file(GLOB_RECURSE CROWDCOUNT_POSTPROCESS ${PROJECT_SOURCE_DIR}/CrowdCountPostProcess/*cpp)
file(GLOB_RECURSE CROWDCOUNT ${PROJECT_SOURCE_DIR}/CrowdCount/*cpp)
set(TARGET crowd_counting)
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME  "$ENV{MX_SDK_HOME}")

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
)

add_executable(${TARGET} main.cpp ${CROWDCOUNT} ${CROWDCOUNT_POSTPROCESS})
target_link_libraries(${TARGET}
        glog
        mxbase
        cpprest
        opencv_world
        )
install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
