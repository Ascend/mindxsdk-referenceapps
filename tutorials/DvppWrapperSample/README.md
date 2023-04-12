
# DvppWrapper接口样例

## 介绍
提供DvppWrapper接口样例，对图片实现编码，解码，缩放，抠图，以及把样例图片编码为264视频文件。

### 1.1 支持的产品

本项目以昇腾Atlas 500 A2为主要的硬件平台。

## 支持的版本

| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

## 准备
打开百度图片https://image.baidu.com/，输入任何关键字，然后搜索，右击任意图片，点击另存为。把图片保存在DvppWrapperSample目录下。

## 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置环境变量
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
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
./DvppWrapperSample ./保存的图片
```
最后会生成缩放resize_result.jpg、抠图后保存的图片write_result_crop.jpg，以及编码保存的视频test.h264。