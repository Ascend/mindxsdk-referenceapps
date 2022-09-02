# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
cmake_minimum_required(VERSION 3.5.2)

project(mxpi_sampleplugin)

if (NOT DEFINED MX_SDK_HOME)
	set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
    message(STATUS "set default MX_SDK_HOME: ${MX_SDK_HOME}")
else ()
    message(STATUS "env MX_SDK_HOME: ${MX_SDK_HOME}")
endif()

add_subdirectory("./src/mxpi_sampleplugin")
