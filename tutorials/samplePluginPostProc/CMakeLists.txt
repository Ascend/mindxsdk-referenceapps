cmake_minimum_required(VERSION 3.6)
project(plugin_sample)

set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")

add_subdirectory(./mxVision/C++)
add_subdirectory(./mindx_sdk_plugin)