# FairMOT目标跟踪

## 1 介绍

FairMOT目标跟踪后处理插件基于MindXSDK开发，在晟腾芯片上进行目标检测和跟踪，可以对行人进行画框和编号，将检测结果可视化并保存。项目主要流程为：通过live555服务器进行拉流输入视频，然后进行视频解码将264格式的视频解码为YUV格式的图片，图片缩放后经过模型推理进行行人识别，识别结果经过FairMOT后处理后得到识别框，对识别框进行跟踪并编号，用编号覆盖原有的类别信息，再将识别框和类别信息分别转绘到图片上，最后将图片编码成视频进行输出。 

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本

本样例配套的CANN版本为[7.0.RC1](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial)。支持的SDK版本为[5.0.RC3](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk)。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

基于MindX SDK的FairMOT目标识别业务流程为：待检测视频存放在live555服务器上经mxpi_rtspsrc拉流插件输入，然后使用视频解码插件mxpi_videodecoder将视频解码成图片，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测结果，本项目开发的FairMOT后处理插件处理推理结果，得到识别框。再接入跟踪插件中识别框进行目标跟踪，得到目标的跟踪编号，然后在使用本项目开发的mxpi_trackidreplaceclassname插件将跟踪编号覆盖类名信息，使用mxpi_object2osdinstances和mxpi_opencvosd分别将识别框和类名（存储跟踪编号）绘制到原图片，再通过mxpi_videoencoder将图片合成视频。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统               | 功能描述                                                     |
| ---- | -------------------- | :----------------------------------------------------------- |
| 1    | 视频输入             | 接收外部调用接口的输入视频路径，对视频进行拉流，并将拉取的裸流存储到缓冲区（buffer）中，并发送到下游插件。 |
| 2    | 视频解码             | 用于视频解码，当前只支持H264/H265格式。                      |
| 3    | 数据分发             | 对单个输入数据分发多次。                                     |
| 4    | 数据缓存             | 输出时为后续处理过程另创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输出到下流插件的数据。 |
| 5    | 图像处理             | 对解码后的YUV格式的图像进行指定宽高的缩放，暂时只支持YUV格式 的图像。 |
| 6    | 模型推理插件         | 目标分类或检测，目前只支持单tensor输入（图像数据）的推理模型。 |
| 7    | 模型后处理插件       | 实现对FairMOT模型输出的tensor解析，获取目标检测框以及对应的ReID向量，传输到跟踪模块。 |
| 8    | 跟踪插件             | 实现多目标（包括机非人、目标）路径记录功能。                 |
| 9    | 跟踪编号取代类名插件 | 用跟踪插件产生的编号信息取代后处理插件产生的类名信息，再将数据传入数据流中。 |
| 10   | 目标框转绘插件       | 将流中传进的MxpiObjectList数据类型转换可用于OSD插件绘图所使用的 MxpiOsdInstancesList数据类型。 |
| 11   | OSD可视化插件        | 主要实现对每帧图像标注跟踪结果。                             |
| 12   | 视频编码插件         | 用于将OSD可视化插件输出的图片进行视频编码，输出视频。        |

### 1.4 代码目录结构与说明

本工程名称为FairMOT，工程目录如下图所示：

```
├── models
│   ├── aipp_FairMOT.config            # 模型转换aipp配置文件
│   ├── mot_v2.onnx      # onnx模型
│   └── mot_v2.om               # om模型
├── pipeline
│   └── fairmot.pipeline        # pipeline文件
├── plugins
│   ├── FairmotPostProcess     # Fairmot后处理插件
│   │   ├── CMakeLists.txt        
│   │   ├── FairmotPostProcess.cpp  
│   │   ├── FairmotPostProcess.h
│   │   └── build.sh
│   └── MxpiTrackIdReplaceClassName  # 跟踪编号取代类名插件
│       ├── CMakeLists.txt
│       ├── MxpiTrackIdReplaceClassName.cpp
│       ├── MxpiTrackIdReplaceClassName.h
│       └── build.sh
├── CMakeLists.txt
├── build.sh
├── main.cpp
└── run.sh
```



### 1.5 技术实现流程图

![](https://gitee.com/seven-day/mindxsdk-referenceapps/raw/master/contrib/FairMOT/image/image1.png)



## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 5.0.RC3       | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| ubuntu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 7.0.RC1       | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |

在编译运行项目前，需要设置环境变量：

```
export MX_SDK_HOME=${SDK安装路径}/mxVision
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径;install_path替换为开发套件包所在路径。LD_LIBRARY_PATH用以加载开发套件包中lib库。



## 3 软件依赖

推理中涉及到第三方软件依赖如下表所示。

| 依赖软件 | 版本       | 说明                           | 使用教程                                                     |
| -------- | ---------- | ------------------------------ | ------------------------------------------------------------ |
| live555  | 1.09       | 实现视频转rstp进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) |
| ffmpeg   | 2021-07-21 | 实现mp4格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md#https://ffmpeg.org/download.html) |



## 4 模型转换

本项目中适用的模型是FairMOT模型，onnx模型可以直接[下载](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/FairMOT/mot_v2.onnx)。下载后使用模型转换工具 ATC 将 onnx 模型转换为 om 模型。
原始ATC样例：https://gitee.com/ascend/samples/tree/master/python/contrib/object_tracking_video

模型转换，步骤如下：

1. 从上述 onnx 模型下载链接中下载 onnx 模型至 `FairMOT/models` 文件夹下，文件名为：mot_v2.onnx 。
2. 进入 `FairMOT/models` 文件夹下执行命令：

```
atc --input_shape="input.1:1,3,608,1088" --check_report=./network_analysis.report --input_format=NCHW --output=./mot_v2 --soc_version=Ascend310 --insert_op_conf=./aipp_FairMOT.config --framework=5 --model=./mot_v2.onnx
```

执行该命令后会在当前文件夹下生成项目需要的模型文件 mot_v2.om。执行后终端输出为：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

表示命令执行成功。



## 5 准备

按照第3小节**软件依赖**安装live555和ffmpeg，按照 [Live555离线视频转RTSP说明文档](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)将mp4视频转换为h264格式。并将生成的264格式的视频上传到`live/mediaServer`目录下，然后修改`FairMOT/pipeline`目录下的fairmot.pipeline文件中mxpi_rtspsrc0的内容。

```
        "mxpi_rtspsrc0": {
            "factory": "mxpi_rtspsrc",
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己所使用的服务器和文件名
                "channelId": "0"
            },
            "next": "mxpi_videodecoder0"
        },
```



## 6 编译与运行

**步骤1** 按照第2小节**环境依赖**中的步骤设置环境变量。

**步骤2** 按照第 4 小节 **模型转换** 中的步骤获得 om 模型文件，放置在 `FairMOT/models` 目录下。

**步骤3** 修改`FairMOT/plugins/FairmotPostProcess`和`FairMOT/plugins/mxpi_trackidreplaceclassname`文件夹下的CMakeLists.txt文件。将其中的"$ENV{MX_SDK_HOME}"修改成自己的SDK目录。

**步骤4** 编译。进入 `FairMOT` 目录，在 `FairMOT` 目录下执行命令：

```
bash build.sh
```

命令执行成功后会在`FairMOT/plugins/FairmotPostProcess`和`FairMOT/plugins/MxpiTrackIdReplaceClassName`目录下分别生成build文件夹。将`FairMOT/plugins/MxpiTrackIdReplaceClassName/build`目录下生成的libmxpi_trackidreplaceclassname.so下载后上传到`${SDK安装路径}/mxVision/lib/plugins`目录下，同时将`FairMOT/plugins/FairmotPostProcess/build`目录下生成的libfairmotpostprocess.so下载后上传到`${SDK安装路径}/mxVision/lib/plugins`目录下。

**步骤5** 运行。回到FairMOT目录下，在FairMOT目录下执行命令：

```
bash run.sh
```

命令执行成功后会在当前目录下生成检测结果视频文件out.h264,查看文件验证目标跟踪结果。

## 7 性能测试

**测试帧率：**

使用`FairMOT/test`目录下的main.cpp替换`FairMOT`目录下的main.cpp，然后按照第6小节编译与运行中的步骤进行编译运行，服务器会输出运行到该帧的平均帧率。

![](https://gitee.com/seven-day/mindxsdk-referenceapps/raw/master/contrib/FairMOT/image/image2.png)

注：输入视频帧率应高于25，否则无法发挥全部性能。

## 8 常见问题

7.1 视频编码参数配置错误

**问题描述：**

`FairMOT/pipeline/fairmot.pipeline`中视频编码分辨率参数目前配置为1280*720。  
该参数通过imageHeight 和 imageWidth 属性配置，且需要和视频输入分配率相同，否则会包如下类型的错：

![](https://gitee.com/seven-day/mindxsdk-referenceapps/raw/master/contrib/FairMOT/image/image3.png)

**解决方案：**

确保`FairMOT/pipeline/fairmot.pipeline`中 mxpi_videoencoder0 插件的 imageHeight 和 imageWidth 属性值是输入264视频的高和宽。
