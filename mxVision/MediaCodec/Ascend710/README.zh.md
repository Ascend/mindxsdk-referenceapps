# Media Codec

## 1.介绍

视频转码样例是基于`mxVision`提供的插件库实现将视频解码、缩放、编码的流程。目前能满足如下的性能：

| 格式 | 路数           |
| - | - |
| **D1**（height: 480 width: 720） | 10 |
| **CIF**（height: 288 width: 352） | 16 |

## 2.环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ------------------------------------ | -------------- |
| Atlas 300I Pro 推理卡 | Ubuntu 18.04 |
| Atlas 300V Pro 推理卡 | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| mxVision | 0.2    |

## 3.预准备

脚本转换为unix格式以及添加脚本执行权限

```bash
chmod +x run.sh
```

## 4.编译

- 配置环境变量

```bash
export MX_SDK_HOME=${安装路径}/mxVision
```

## 5.运行

### 5.1 运行前配置

- 构建rtsp视频流服务

配置文件配置项说明

```bash
可通过修改配置文件(setup.config)stream.deviceId配置项，设置device id值
stream.deviceId = 0

可通过修改配置文件(setup.config)stream.channelCount配置项，设置视频拉流路数
stream.channelCount = 1

可通过修改配置文件(setup.config)stream.channelCount配置项，设置是否打印fps
stream.fpsMode = 1

可通过修改配置文件(setup.config)stream.channelCount配置项，设置视频拉流地址
stream.ch0 = rtsp://xxx.xxx.xxx.xxx:xxx/xxx.264

注:stream.ch最小数量需不小于stream.channelCount传入数值

```

转码过程主要是：`视频拉流`--》`视频解码`--》`图像缩放`--》`视频编码`，根据用户要求修改芯片deviceId、rtsp视频流地址、vdecChannelId、缩放大小（**D1**/**CIF**）等。


**注意**：解码模块`mxpi_videodecoder`的**vdecChannelId**配置项要保证不重用；缩放模块`mxpi_imageresize`的**resizeHeight**和**resizeWidth**要与编码模块的`mxpi_videoencoder`的**imageHeight**和**imageWidth**保持一致；`mxpi_videoencoder`编码模块的**fps**用于控制是否打印帧率，默认值是**0**表示不打印，若要打印，可设置为**1**;**deviceId**配置为需要运行的npu芯片编号，具体可以通过`npu-smi info`查看。

- 修改MindXSDK的日志配置文件

参考mxVision用户指南D.2章节，修改`${MX_SDK_HOME}/mxVision/config/logging.conf`，调节输出日志级别为info级别。

```bash
# will output to stderr, where level >= global_level，default is 0
# Log level: -1-debug, 0-info, 1-warn, 2-error, 3-fatal。
global_level=0
```

### 5.2 运行

```bash
bash run.sh
```
