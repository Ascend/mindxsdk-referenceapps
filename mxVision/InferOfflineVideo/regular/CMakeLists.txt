cmake_minimum_required(VERSION 3.10)
project(InferOfflineVideo)

add_compile_options(-std=c++11 -pthread -fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z,noexecstack -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(OUTPUT_NAME "main")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set(MX_SDK_HOME ${SDK°²×°Â·¾¶})

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /xxx/Ascend/ascend-toolkit/5.0.3/arm64-linux/runtime/lib64/stub
)

add_executable(${OUTPUT_NAME} main.cpp)
target_link_libraries(${OUTPUT_NAME}
        glog
        mxbase
        streammanager
        cpprest
        mindxsdk_protobuf
        )

