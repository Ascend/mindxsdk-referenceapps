# 1 Cmake介绍
CMake（跨平台编译，Cross platform Make。）是一个跨平台的自动化建构系统,它使用一个名为 CMakeLists.txt 的文件来描述构建过程,可以产生标准的构建文件,如 Unix 的 Makefile 或Windows Visual C++ 的 projects/workspaces 。它能够输出各种各样的makefile或者project文件,能测试编译器所支持的C++特性,类似UNIX下的automake。

## 1.1 Cmake使用方法
CMake的所有的语句都写在一个叫:CMakeLists.txt的文件中。当CMakeLists.txt文件确定后,可以用cmake命令对相关的变量值进行配置。这个命令必须指向CMakeLists.txt所在的目录。配置完成之后,应用cmake命令生成相应的makefile。

## 1.2 Cmake基本构建流程

Cmake 构建文件，用于存储构建系统文件（比如makefile以及其他一些cmake相关配置文件）和构建输出文件（编译生成的中间文件、可执行程序、库）的顶级目录。它生成编译需要的makefile及其他中间文件，再用make构建工程，获得可执行的目标文件。构建方式包括以下两种：

1. 内部构建
   在目录下创建源文件和CMakeLists.txt文件，执行cmake . .代表本目录，生成CMakeFiles，CMakeCache.txt，cmake_install.cmake，Makefile等文件，进入makefile，在Makefile目录执行make命令，构建实际工程，得到目标文件。

2. 外部构建（推荐）
   新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/workspace/cmake-examples/01-basic/C-static-library/build

make
[ 50%] Building CXX object CMakeFiles/main.dir/main.cpp.o
[100%] Linking CXX executable ../main
[100%] Built target main

```

##
在使用IDE进行开发时，在编写好CMakeList.txt文件后，IDE会自动生成相应的外部构建目录，所有临时构建和目标文件都位于此目录中，保持源代码树的整洁。

Clion的CMake构建目录为：cmake-build-debug目录  
![13.png](img/1622265242971.png '13.png')
      
这就是CMake的执行，其难点在于如何编写CMakeLists.txt文件。

## 1.3 语法
关于CMake，首先要编写CMake构建文件(CMakeLists.txt和*.cmake)，它是由一些列命令构成，需要掌握一些常用的命令，包括变量定义、条件块、循环块、宏定义和函数定义，这些定义本身也是命令。CMakeLists.txt 的语法比较简单，由命令、注释和空格组成，其中命令是不区分大小写的。符号#后面的内容被认为是注释。命令由命令名称、小括号和参数组成，参数之间使用空格进行间隔。
1. 注释
Cmake用”#”注释，"#"后面为注释的内容，从"#"字符开始到此行结束。CMake>=3.0的时候支持多行注释，以#[[ 开始进行块注释，并且在块注释的一端与 ]]结束。
2. 变量定义和引用
使用set命令显式定义及赋值，在非if语句中，使用${<variable>}引用，if中直接使用变量名引用；后续的set命令会清理变量原来的值；
  ```
  set(var a;b;c) <=> set(var a b c)  #定义变量var并赋值为a;b;c这样一个string list
  Add_executable(${var}) <=> Add_executable(a b c)  
  ```
  在引用参数和非引用参数中使用，它的引用形式是${<variable>}。
  Cmake调用环境变量方式：使用$ENV{NAME}指令就可以调用系统的环境变量了。
  ```
  比如 MESSAGE(STATUS “HOME dir: $ENV{HOME}”) 设置环境变量的方式是：SET(ENV{变量名} 值)
  ```
3. 条件语句
  ```
  if(var)        #var 非empty 0 N No OFF FALSE... 非运算使用NOT
  …
  else()/elseif()
  …
  endif(var)
  ```
4. 循环语句
  ```
  Set(VAR a b c)
  Foreach(f ${VAR})  …Endforeach(f)
  WHILE() … ENDWHILE()
  ```

## 1.4 常用系统变量和命令
  该小结我们提供了以一些CMake常用的变量和命令供大家参考学习：
  [CMake的常用变量](https://bbs.huaweicloud.com/forum/thread-117324-1-1.html)
  [CMake的常用命令](https://bbs.huaweicloud.com/forum/thread-117221-1-1.html)
  如果想对CMake有更进一步的了解的话，也可以阅读[CMake官方文档](https://cmake.org/cmake/help/v3.20/)进行深入学习。

## 1.5 CMakeLists文件示例
  以图像检测样例的CMakeLists文件为例。

  ## 
  **前提条件**：
  参考[IDE开发环境搭建](./1-2IDE开发环境搭建.md)章节，将本地IDE与远程环境连接起来，并同步好项目文件。

  ##
 编写CMakeLists文件，编写时请参考以下内容：
  ```
# 最低CMake版本
cmake_minimum_required(VERSION 3.14)

# 项目名
project(test)

# 设置编译选项
add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
  
  
# 配置环境变量MX_SDK_HOME，如：/home/xxxxxxx/MindX_SDK/mxVision,可在远程环境中用指令env查看
set(MX_SDK_HOME ${用户自己的SDK安装路径})
# 设置所需变量
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

# 链接项目所需头文件路径
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)

# 链接项目所需库文件路径
link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        )


# 生成main.cpp的可执行文件sample
add_executable(sample main.cpp)

# 把所需库与可执行文件main链接起来
target_link_libraries(sample
        glog
        mxbase
        mxpidatatype
        plugintoolkit
        streammanager
        cpprest
        mindxsdk_protobuf
        opencv_world)
  ```
 ##
  修改好CMakeLists文件后，重新加载CMakeLists文件。  
  ![1.png](img/1622518642593.png '1.png')

##
至此，样例项目的CMakeLists文件配置完成，可以对项目进行运行调试。









