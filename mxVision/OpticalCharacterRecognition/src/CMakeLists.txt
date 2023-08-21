# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
cmake_minimum_required(VERSION 3.10.2)
project(OcrDemo)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT/CMake})
set(CMAKE_SKIP_BUILD_RPATH True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT/dist})
add_compile_options(-DENABLE_DVPP_INTERFACE)


set(AscendBaseFolder ${PROJECT_SRC_ROOT/AscendBase/Base})

# Find AscendBase
set(ASCEND_BASE_DIR ${AscendBaseFolder})
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)

file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
        ${ASCEND_BASE_ABS_DIR}/BlockingQueue/*cpp\
        ${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleManagers/*cpp
        ${ASCEND_BASE_ABS_DIR}/ArgumentParser/*cpp
        )

# Find Header
set(HEADER_FILE_DIR
        ${ASCEND_BASE_ASB_DIR}
        ${ASCEND_BASE_ASB_DIR}/Framework
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors/CharacterRecognitionPost
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors/TextDetectionPost
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors/ModuleProcessors
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors/Signal
        ${ASCEND_BASE_ASB_DIR}/Framework/ModuleProcessors/Utils
        )
#include_directories(${HEADER_FILE_DIR})

# Find Src
file(GLOB_RECURSE SRC_FILE
        ${PROJECT_SRC_ROOT}/main.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/HandOutProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/DbnetPreProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/DbnetInferProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/DbnetPostProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/ClsPreProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/ClsInferProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/ClsPostProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/CrnnPreProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/CrnnInferProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/CrnnPostProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/CollectProcess/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/CharacterRecognitionPost/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/TextDetectionPost/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/Signal/*.cpp
        ${ASCEND_BASE_ABS_DIR}/Framework/ModuleProcessors/Processors/Utils/*.cpp
        )

set(SOURCE_FILE
        ${SRC_FILE}
        ${ASCEND_BASE_SRC_FILES}
        )

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH_PATTERN $ENV{ARCH_PATTERN})
include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv7
        ${MX_SDK_HOME}/include/MxBase
        ${ASCEND_HOME_PATH}/${ARCH_PATTERN}/include
        ${ASCEND_HOME_PATH}/${ARCH_PATTERN}/runtime/include
)

link_directoried(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/lib
        ${ASCEND_HOME_PATH}/atc/lib64
)

# Set the target executable file
add_executable(main ${SOURCE_FILE})

target_link_libraried(main ascendcl acl_dvpp opencv_world mxbase cpprest glog pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)