cmake_minimum_required(VERSION 3.10.2)
project(main)

set(TARGET main)

add_definitions(-D_GLIBCXX_USE_CXX_ABI=0)
add_definitions(-DENABLE_DVPP_INTERFACE)

add_compile_options(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++17 -fPIE -fstack-protector-all -fPIC -Wall)
add_compile_options(-g)

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

set(TASKFLOW_PATH ${CMAKE_SOURCE_DIR}/include/)

set(FFMPEG_PATH /usr/local/ffmpeg)
set(FFMPEG_LIB_PATH /usr/local/ffmpeg/lib)
set(FFMPEG_LIBRARIES ${FFMPEG_LIB_PATH}/libavcodec.so;${FFMPEG_LIB_PATH}/libavdevice.so;${FFMPEG_LIB_PATH}/libavfilter.so;${FFMPEG_LIB_PATH}/libavformat.so;${FFMPEG_LIB_PATH}/libavutil.so;${FFMPEG_LIB_PATH}/libswresample.so;${FFMPEG_LIB_PATH}/libswscale.so)

include_directories(/usr/local/ffmpeg/include)
include_directories(${TASKFLOW_PATH})

set(MXBASE_INC ${MX_SDK_HOME}/include)
set(MXBASE_LIB_DIR ${MX_SDK_HOME}/lib)
set(MXBASE_POST_LIB_DIR ${MX_SDK_HOME}/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_INC ${MX_SDK_HOME}/include/MxBase/postprocess/include)
set(OPENSOURCE_DIR ${MX_SDK_HOME}/opensource)

set(ACL_INC_DIR ${ASCEND_HOME_PATH}/acllib/include)
set(ACL_LIB_DIR ${ASCEND_HOME_PATH}/acllib/lib64)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--copy-dt-needed-entries")

include_directories(${ACL_INC_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)

include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_INC})

include_directories(${MX_SDK_HOME}/aarch64-linux/include/)

link_directories(${ACL_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR)

    
link_directories(../../../opencv4.5.5/lib64)

find_package(Threads REQUIRED)

add_subdirectory(postprocessor)
add_subdirectory(utils)

add_executable(main main.cpp)
target_link_libraries(main 
            glog 
            mxbase 
            cpprest 
            yolov3postprocess 
            resnetAttributePostProcess 
            SsdVggPostProcess 
            carPlateRecognitionPostProcess 
            Threads:Threads
            ${FFMPEG_LIBRARIES}
            opencv_world
            multiObjectTracker
            objectSelection
            faceAlignment
            faceLandMark
)

install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)