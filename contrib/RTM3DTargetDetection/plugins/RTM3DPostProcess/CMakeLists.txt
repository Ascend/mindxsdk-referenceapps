# 一般写在根目录第一行，表示当前项目工程要求的CMake的最低版本
cmake_minimum_required(VERSION 3.5.2)

# 声明项目名
project(rtm3dpostprocess)

# 定义变量，特别说明的是，可以定义系统变量
set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
set(PLUGIN_NAME "rtm3dpostprocess")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(LIBRARY_OUTPUT_PATH ../../plugin)    # 输出路径

# 链接库函数指令,链接项目所需头文件路径
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

# 链接项目所需库文件路径
link_directories(${MX_SDK_HOME}/opensource/lib)
link_directories(${MX_SDK_HOME}/lib)

# 设置编译选项
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

# shared为动态库，改成static则生成静态库
# add_library(test shared test.cpp) 根据test.cpp生成共享库文件test.so，使用的时候用库文件名test来引用test.so
# 编译共享库
add_library(${TARGET_LIBRARY} SHARED RTM3DPostProcess.cpp RTM3DPostProcess.h)

# 把所需库与可执行文件${TARGET_LIBRARY}链接起来
target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0 glog)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase)
