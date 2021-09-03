
# DvppWrapper接口样例

## 介绍
提供DvppWrapper接口样例，对图片实现编码，解码，缩放，抠图，以及把样例图片编码为264视频文件。

## 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置环境变量
ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径
```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** cd到DvppWrapperSample目录下，执行如下编译命令：
```
mkdir build
cd build
cmake ..
make
```

**步骤4** cd到DvppWrapperSample目录下，可看到可执行文件DvppWrapperSample， 实行命令：
```
./DvppWrapperSample
```
最后会生成缩放、抠图后保存的图片，以及编码保存的视频。