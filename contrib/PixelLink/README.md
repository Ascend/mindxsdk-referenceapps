# Pixellink文本检测

## 1 介绍
（项目的概述，包含的功能）
（项目的主要流程）

### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本，列出版本号查询方式。

支持的SDK版本为2.0.2。

版本号查询方法，在Atlas产品环境下，运行命令：npu-smi info进行查看。


### 1.3 软件方案介绍

请先总体介绍项目的方案架构。如果项目设计方案中涉及子系统，请详细描述各子系统功能。如果设计了不同的功能模块，则请详细描述各模块功能。

本系统设计了不同的功能模块。主要流程为：图片传入流中，利用Yolov4的检测模型检测人脸，将检测出人脸的图像放缩至特定尺寸，再利用提取关键点的模型进行关键点提取，获取人脸关键点后进行对齐，将对齐结果输入到人脸属性识别模型中，最后以序列化方式输出识别结果。各模块功能描述如表1.1所示：

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | XXX    | 详细功能描述 |
| 2    | XXX    | 详细功能描述 |

表1.2 系统方案中各模块功能：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | XXX    | 详细功能描述 |
| 2    | XXX    | 详细功能描述 |



### 1.4 代码目录结构与说明

本工程名称为PixelLink，工程目录如下图所示：



### 1.5 技术实现流程图

（[Pixellink文本检测流程图](https://images.gitee.com/uploads/images/2021/1029/112024_3a19c293_9366121.png "屏幕截图.png")





## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称  | 版本   |
| -------- | ------ |
| cmake    | 3.5+   |
| mxVision | 2.0.2  |
| python   | 3.7.5  |

在编译运行项目前，需要设置环境变量：

模型转换所需ATC工具环境搭建参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0004.html

在编译运行项目前，需要设置环境变量：

步骤1：安装mxVision SDK。 
步骤2：配置mxVision SDK环境变量、lib库环境变量以及python环境变量。

```
export MX_SDK_HOME=${安装路径}/mxVision
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins"
```
环境变量介绍
MX_SDK_HOME为SDK安装路径
LD_LIBRARY_PATH为lib库路径
PYTHONPATH为python环境路径


## 依赖安装

（依赖搭建安装或者获取方式的具体步骤）



## 4 编译与运行
（描述项目安装运行的全部步骤，，如果不涉及个人路径，请直接列出具体执行命令）

示例步骤如下：
**步骤1** （修改相应文件）

**步骤2** （设置环境变量）

**步骤3** （执行编译的步骤）

**步骤4** （运行及输出结果）


## 5 常见问题

请按照问题重要程度，详细列出可能要到的问题，和解决方法。

### 6.1 XXX问题

**问题描述：**

截图或报错信息

**解决方案：**

详细描述解决方法。