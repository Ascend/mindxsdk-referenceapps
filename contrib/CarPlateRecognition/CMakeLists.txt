cmake_minimum_required(VERSION 3.10) #cmake最低版本要求为3.10
project(Car_Plate_Recognition) #项目工程名

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*cpp ) #全局递归寻找指定路径下的所有.cpp文件
set(TARGET car_plate_recognition)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall) #添加编译选项
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

#设置SDK的安装路径
set(MX_SDK_HOME "/home/zhongzhi3/MindX_SDK/mxVision")

#添加.h文件的搜索路径
include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${PROJECT_SOURCE_DIR}/include
        /home/zhongzhi3/freetype-2.10.0/include
)

#添加lib的搜索路径
link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
        /usr/local/Ascend/driver/lib64/
        )

#生成可执行文件car_plate_recognition.exe
add_executable(car_plate_recognition ${SRC_FILES})

#将target链接到lib
target_link_libraries(car_plate_recognition
        glog #${MX_SDK_HOME}/opensource/lib下有个libglog.so
        mxbase #${MX_SDK_HOME}/lib下有个libmaxbase.so
        cpprest #${MX_SDK_HOME}/opensource/lib下有个libcpprest.so
        opencv_world #${MX_SDK_HOME}/opensource/lib下有个libopencv_world.so
        freetype
        )

#指定exe文件的输出路径
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
