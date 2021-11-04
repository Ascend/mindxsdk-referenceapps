cmake_minimum_required(VERSION 3.10)
project(edge_detection_picture)

include_directories(./rcfPostProcess)
include_directories(./rcfDetection)
file(GLOB_RECURSE RCF_POSTPROCESS ${PROJECT_SOURCE_DIR}/rcfPostProcess/*cpp)
file(GLOB_RECURSE RCF_DETECTION ${PROJECT_SOURCE_DIR}/rcfDetection/*cpp)
set(TARGET edge_detection_picture)
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
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
        /usr/local/Ascend/driver/lib64/
)

add_executable(edge_detection_picture main.cpp ${RCF_DETECTION} ${RCF_POSTPROCESS})
target_link_libraries(edge_detection_picture
        glog
        mxbase
        cpprest
        opencv_world
	boost_system
	boost_filesystem
        )
install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
