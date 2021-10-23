# C++ 基于MxBase的车流量统计开发
## 1 介绍
车流量统计是指对视频中的车辆进行计数，实现对本地视频（H264）进行车辆追踪并计数，最后生成可视化结果。车流统计分为五个步骤：车辆视频流读取、车辆检测、车辆追踪、车辆计数以及结果可视化。
### 1.1 支持的产品
支持昇腾310芯片
### 1.2 支持的版本
本样例配套的CANN版本为3.3.0，MindX SDK版本为2.0.2
### 1.3 软件方案介绍
车流统计项目实现：输入类型是视频数据（需要将视频转换为.264的视频格式），ffmpeg打开视频流获取视频帧信息，图像经过尺寸大小变换，满足模型的输入尺寸要求；将尺寸变换后的图像数据依次输入Yolov4检测模型进行推理，模型输出经过后处理后，使用SORT算法进行车辆追踪得到车辆轨迹，再设置标志对车辆进行计数，最后得到某时刻已经通过的车辆数。

本流程的视频检测模块参考的是Ascend的参考样例：[https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/mxBaseVideoSample](http://)

表1.1 系统方案中各模块功能：
| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 设备初始化        | 调用mxBase::DeviceManager接口完成推理卡设备的初始化。        |
| 2    | 视频输入          | 使用ffmpeg打开视频流获取视频帧信息                                        |
| 3    | 图像解码/图像缩放 | 调用mxBase::DvppWrapper.DvppVdec()函数完成图像解码，VpcResize()完成缩放。 |
| 4    | 模型推理/后处理   | 调用mxBase:: ModelInferenceProcessor 接口完成车辆目标检测模型推理并进行后处理  |
| 5    | 目标跟踪         | 调用mxBase::MultipleObjectTracking和Hungarian接口完成车辆目标跟踪        |
| 6    | 车辆计数         | 设置标志对通过的车辆进行计数                                |
| 7    | 保存结果         | 使用opencv进行结果可视化并保存为视频文件                      |
| 8    | 资源释放         | 调用mxBase::DeviceManager接口完成推理卡设备的去初始化。      |

### 1.4 代码目录结构与说明

本sample工程名称为VehicleCounting，工程目录如下图所示：
```
.
├── data
│   └── test.264
├── model
│   ├── aipp_yolov3_416_416.aippconfig
│   ├── coco.names
│   ├── yolov3_tf_bs1_fp16.om
│   ├── yolov4_bs.om
├── BlockingQueue
│   ├── BlockingQueue.h
├── VideoProcess
│   ├── DataType.h
│   ├── Hungarian.h
│   ├── Hungarian.cpp
│   ├── KalmanTracker.h
│   ├── KalmanTracker.cpp
│   ├── MOTConnection.h
│   ├── MOTConnection.cpp
│   ├── VideoProcess.h
│   ├── VideoProcess.cpp
├── Yolov4Detection
│   ├── Yolov4Detection.h
│   ├── Yolov4Detection.cpp
├── CMakeLists.txt
├── main.cpp
├── README.md
└── run.sh
```



### 1.5 技术实现流程图

![Image text](https://gitee.com/wu-jindge/mindxsdk-referenceapps/raw/master/contrib/VehicleCounting/img/process.JPG)

## 2 环境依赖
环境依赖软件和版本如下表：



| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.2        | mxVision软件包                | [链接](https://www.hiascend.com/software/mindx-sdk/mxvision) |
| Ascend-CANN-toolkit | 3.3.0        | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |
| ffmpeg             | 4.2.1        | 视频转码解码组件              | [安装教程](https://bbs.huaweicloud.com/forum/thread-142431-1-1.html)|                                              
| ffmpeg             | 4.2.1        | 视频转码解码组件              | [安装教程](https://bbs.huaweicloud.com/forum/thread-142431-1-1.html)| 
| pc端ffmpeg         | 2021-09-01   | 将视频文件格式转换为.264      | [安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md)|

## 3 模型转换
