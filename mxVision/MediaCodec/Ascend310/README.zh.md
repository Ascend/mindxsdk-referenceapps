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
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡（型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |

支持的SDK版本为 5.0.RC1, CANN 版本310使用6.3.RC1，310B使用6.2.RC1。

## 3.预准备

脚本转换为unix格式

```bash
sed -i 's/\r$//' ./script/*.sh
```

给script目录下的脚本添加执行权限

## 4.编译

- 配置环境变量

```bash
export MX_SDK_HOME=${安装路径}/mxVision
```

- 执行`./script/build.sh`，在dist文件夹中会生成`mxVisonCodec`。

```bash
./script/build.sh
```

## 5.运行

### 5.1 运行前配置

- 构建rtsp视频流服务

构建rtsp视频流服务生成rtsp流地址。

- 修改pipeline

转码过程主要是：`视频拉流`--》`视频解码`--》`图像缩放`--》`视频编码`，根据用户要求修改芯片deviceId、rtsp视频流地址、vdecChannelId、缩放大小（**D1**/**CIF**）等。

为了提升修改效率，提供了`./script/create_pipeline.sh`脚本，需要修改脚本里面的配置参数。

```bash
#!/bin/bash
file_path=$(cd `dirname $0`; pwd)

#需要生成的多少路pipelibe
channel_nums=xxx

#每路pipeline的rtsp流地址，数组长度跟${channel_nums}一致，请确保rtsp地址存在。
#若使用live555推流工具构架rtsp流，rstp流格式为rtsp://${ip_addres}:${port}/${h264_file}
#${ip_addres}：起流的机器ip地址。如果是本地起流可以设置为127.0.0.1；如果是其他机器起流，那需要配置该台机器的ip地址
#${port}：可使用的rtsp流的端口
#${h264_file}:需要推流的h264视频文件，一般都是以.264结尾的文件
rtsp_array=(xxx xxx xxx)

#配置pipeline运行的npu编号
device_id=xxx

#输出图像尺寸. CIF(height: 288 width: 352),D1(height: 480 width: 720)
height=xxx
width=xxx

#是否打印转码的帧率. 0：不打印，1：打印
fps=xxx

#I帧间隔.一般设置视频帧率大小，25或者30
i_frame_interval=xxx
```

执行脚本，生成的pipeline文件在`./pipeline/`目录下，文件名类似`testxxx.pipeline`。

```
./script/create_pipeline.sh
```

**注意**：解码模块`mxpi_videodecoder`的**vdecChannelId**配置项要保证不重用；缩放模块`mxpi_imageresize`的**resizeHeight**和**resizeWidth**要与编码模块的`mxpi_videoencoder`的**imageHeight**和**imageWidth**保持一致；`mxpi_videoencoder`编码模块的**fps**用于控制是否打印帧率，默认值是**0**表示不打印，若要打印，可设置为**1**;**deviceId**配置为需要运行的npu芯片编号，具体可以通过`npu-smi info`查看。

- 修改MindXSDK的日志配置文件

参考mxVision用户指南D.2章节，修改`${MX_SDK_HOME}/mxVision/config/logging.conf`，调节输出日志级别为info级别。

```bash
# will output to stderr, where level >= global_level，default is 0
# Log level: -1-debug, 0-info, 1-warn, 2-error, 3-fatal。
global_level=0
```

### 5.2 运行

- 根据实际情况选择运行的转码路数，每路输出的日志会重定向到`./logs/output*.log`中。

```bash
./script/run.sh ${nums}
```

**注意：**调用`./script/run.sh`长时间跑生成的日志可能写爆磁盘。

- 显示每路运行情况

```bash
./script/show.sh
#可以使用定时打印结果(可选)
watch -n 1 ./script/show.sh
```

- 显示结果

**显示转码帧率**

若设置了输出转码的帧率，会额外输出如下结果，可用于观察性能。若打印的fps和视频的帧率保持一致，说明性能满足要求。否则，说明性能达不到要求。

```bash
 Plugin(mxpi_videoencoder*) in fps (xx)
```

**显示每帧结果（可选）**

显示每一路每一帧的编码输出结果（需要修改日志级别为debug级别）。**mxpi_videodencoder***编码模块输出**frame id**为xxx 数据长度**stream size**为xxxx。

```
Plugin(mxpi_videodencoder*) encode frameId(xxxx) stream size(xxxx)
```

- 停止运行

```
./script/stop.sh
```

**注意**：受限于ACL接口约束，只能实现多进程多路视频转码，run.sh脚本实际上是启动了多个进程实现多路转码，每一路对应一个进程。