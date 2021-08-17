# 人体语义分割的C++实现demo

## 介绍

提供C++版本的人体语义分割样例，实现对传入图片进行人体检测检测，生成对应的可视化语义分割结果图。

### 准备工作

> 环境配置



**步骤1** 根据官网指示下载好MindXSDK的开发环境

**步骤2** 配置好对应的环境变量

### 配置环境变量

```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

### 配置SDK路径

配置CMakeLists.txt文件中的`MX_SDK_HOME`环境变量

```
set(MX_SDK_HOME ${SDK安装路径}/mxVision)
```
### 传入照片
将选中的照片传入/data文件夹下
打开根目录下的main.cpp文件，将主程序中    std::string inputPicname="test.jpeg";的名称改为传入的图像名称


### 编译项目文件

新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

make
Scanning dependencies of target sample
[ 50%] Building CXX object CMakeFiles/sample.dir/main.cpp.o
[100%] Linking CXX executable ../sample
[100%] Built target sample
# sample就是CMakeLists文件中指定生成的可执行文件。
```

### 执行脚本

执行run.sh脚本前请先确认可执行文件sample已生成。

```
chmod +x run.sh
bash run.sh
```

### 查看结果

执行run.sh完毕后，sample会将人体语义分割检测结果保存在工程目录下result文件夹中命名为'reuslt_inpuitPicname.jpg'。