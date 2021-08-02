cmake_minimum_required(VERSION 3.6)
project(plugin_sample)

#该语句中%MX_SDK_HOME%根据实际SDK安装位置修改，可通过在终端运行env命令查看
#set(ENV{MX_SDK_HOME} %MX_SDK_HOME%)

add_subdirectory(./mxVision/C++)
add_subdirectory(./mindx_sdk_plugin)